import os

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
import torch.nn.functional as F
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch.nn.functional as F
import random
from itertools import product
from collections import defaultdict

################
##################建立双向字典以方便查询图像点的位置
class BiDict:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}

    def add(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key)

    def get_key(self, value):
        return self.value_to_key.get(value)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=256, stride=32):
    if panorama_height/8 >= window_size:
        panorama_height /= 8
        panorama_width  /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
    else:
        num_blocks_height = 1
        num_blocks_width  = 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    views_bidi = BiDict()
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        h_num = int((i // num_blocks_width))
        w_num = int((i % num_blocks_width))
        views_bidi.add((h_num, w_num), (h_start, h_end, w_start, w_end))
        views.append((h_start, h_end, w_start, w_end))
    return views,views_bidi,(h_num,w_num)


class GCDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            # model_key = "stabilityai/stable-diffusion-2-1-base"
            model_key = "/mnt/data/project_kyh/weight/stabilityai/stable-diffusion-2-1"
        elif self.sd_version == '2.0':
            model_key = "/mnt/data/project_kyh/weight/stabilityai/stable-diffusion-2"
        elif self.sd_version == '1.5':
            model_key = "/mnt/data/project_kyh/weight/lllyasviel/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.vae.enable_tiling()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.dist_mse_map = {}
        self.dist_mse_2_map = {}
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048,
                        num_inference_steps=50, guidance_scale=7.5,
                        timesteps=[20], scales=[1],
                        upsample_st="bilinear", small_height=256,
                        small_width=1024, cover_ritio=0.75):
        
        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts

        text_embeds = self.get_text_embeds(prompts, negative_prompts)
        latent, h_l, w_l = self.init_latent(height, width)
        latent_small,h_s,w_s = self.init_latent_small(small_height, small_width)
        views, views_bidi, (h_max, w_max) = get_views(height, width)

        self.scheduler.set_timesteps(num_inference_steps)

        # 预先创建 count/value 张量用于复用
        count_buffer = torch.empty_like(latent)
        value_buffer = torch.empty_like(latent)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                latent = self.step_denoise_large(latent, views, text_embeds, t, guidance_scale,
                                                count_buffer, value_buffer)
                latent_small = self.step_denoise_small(latent_small, text_embeds, t, guidance_scale)
                latent = self.inject_hdg(latent, latent_small, i, timesteps, scales, cover_ritio, h_l, w_l, h_s, w_s)


        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())

        imgs_small = self.decode_latents(latent_small)  
        img_small = T.ToPILImage()(imgs_small[0].cpu())
        return img, img_small

    @torch.no_grad()
    def init_latent(self, height, width):
        h = height // 8
        w = width // 8
        latent = torch.randn((1, self.unet.in_channels, h, w), device=self.device)
        return latent, int(h), int(w)

    @torch.no_grad()
    def init_latent_small(self, small_height, small_width):
        h = small_height // 8
        w = small_width // 8
        latent=torch.randn((1, self.unet.in_channels, h, w), device=self.device)
        return latent, int(h), int(w)

    @torch.no_grad()
    def step_denoise_large(self, latent, views, text_embeds, t, guidance_scale,
                        count, value):
        # 使用 fill_() 清空原位张量，避免创建新张量
        count.zero_()
        value.zero_()

        for h_start, h_end, w_start, w_end in views:
            latent_view = latent[:, :, h_start:h_end, w_start:w_end]
            latent_input = torch.cat([latent_view] * 2)
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=text_embeds)['sample']
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']

            value[:, :, h_start:h_end, w_start:w_end] += denoised
            count[:, :, h_start:h_end, w_start:w_end] += 1

        # 避免除以0，同时融合为新 latent
        return torch.where(count > 0, value / count, value)

    @torch.no_grad()
    def step_denoise_small(self, latent_small, text_embeds, t, guidance_scale):
        latent_input = torch.cat([latent_small] * 2)
        noise_pred = self.unet(latent_input, t, encoder_hidden_states=text_embeds)['sample']
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return self.scheduler.step(noise, t, latent_small)['prev_sample']

    @torch.no_grad()
    def inject_hdg(self, latent, latent_small, step_index, timesteps, scales, cover_ritio, h_l, w_l,h_s, w_s):
        count = torch.ones_like(latent)
        value = latent.clone()
        alpha=1
        # 第一阶段：计算每个区域在不同尺度的 dist_mse
        for num, (t_cutoff, scale) in enumerate(zip(timesteps, scales)):
            if step_index >= t_cutoff:
                continue

            window = int(scale * cover_ritio)

            for h in range(0, h_l, scale):
                for w in range(0, w_l, scale):
                    # 计算 patch latent 和 dist_mse
                    patch,dist_mse,dist_mse_2  = self.getwindow_latent(latent,latent_small, h, h_l, h_s, w, w_l, w_s, window, scale)
                    # 保存当前 patch 的 dist_mse
                    self.dist_mse_map[(h, w, scale)] = dist_mse
                    self.dist_mse_2_map[(h, w, scale)] = {"mse": dist_mse,
                                                          "alpha":None
                                                          }

        # 第二阶段：根据 dist_mse 跳过不必要的区域
        skip_map = {}  # 用来记录哪些位置在较大尺度已经跳过，避免在子尺度再执行
        for num, (t_cutoff, scale) in enumerate(zip(timesteps, scales)):
            if step_index >= t_cutoff:
                continue
            window = int(scale * cover_ritio)
            for h in range(0, h_l, scale):
                for w in range(0, w_l, scale):
                    patch,dist_mse,dist_mse_2 = self.getwindow_latent(latent,latent_small, h, h_l, h_s, w, w_l, w_s, window, scale)
                    # 如果已经跳过这个位置，则跳过
                    if (h, w, scale) in skip_map and skip_map[(h, w, scale)]:
                        continue

                    # 若不是最细层级，检查子区域的平均 dist_mse
                    finer_scale = scale // 2
                    if finer_scale >= min(scales):  # 有更小层级可以参考
                        child_keys = [
                            (h + dh, w + dw, finer_scale)
                            for dh, dw in product([0, finer_scale], repeat=2)
                        ]
                        child_mses = [self.dist_mse_map.get(k) for k in child_keys if k in self.dist_mse_map]
                        if len(child_mses) == 4:
                            avg_child_mse = sum(child_mses) / 4
                            if dist_mse > avg_child_mse:
                                skip_map[(h, w, scale)] = True
                                del self.dist_mse_2_map[(h, w, scale)]
                            else:
                                for dh, dw in product([0, finer_scale], repeat=2):
                                    skip_map[(h + dh, w + dw, finer_scale)] = True
                                del self.dist_mse_2_map[(h + dh, w + dw, finer_scale)]

        # # 按 dist_mse 从大到小排序（越大表示质量越差）
        # sorted_items = sorted(self.dist_mse_2_map.items(), key=lambda x: x[1]["mse"], reverse=True)

        # # 获取排序后的元素个数
        # n = len(sorted_items)

        # # 分配 α 值
        # for i, (key, _) in enumerate(sorted_items):
        #     if i < n // 3:
        #         self.dist_mse_2_map[key]["alpha"] = 0.5  # 最优区域，减小引导
        #     elif i < 2 * n // 3:
        #         self.dist_mse_2_map[key]["alpha"] = 1.0  # 中等区域，保持引导
        #     else:
        #         self.dist_mse_2_map[key]["alpha"] = 1.5  # 最差区域，增强引导

        # 按 scale 分组，并在每组内按 mse 排序后分配 alpha

        # scale_groups = defaultdict(list)
        # for key, val in self.dist_mse_2_map.items():
        #     _, _, scale = key
        #     scale_groups[scale].append((key, val["mse"]))

        # for scale, items in scale_groups.items():
        #     # 按 mse 从小到大排序（越小质量越好）
        #     sorted_items = sorted(items, key=lambda x: x[1])
        #     n = len(sorted_items)
        #     for i, (key, _) in enumerate(sorted_items):
        #         if i < n // 3:
        #             self.dist_mse_2_map[key]["alpha"] = 0.9  # 最优区域，增强引导
        #         elif i < 2 * n // 3:
        #             self.dist_mse_2_map[key]["alpha"] = 0.7  # 中等区域，保持引导
        #         else:
        #             self.dist_mse_2_map[key]["alpha"] = 0.5  # 最差区域，减小引导

        alpha_max = 0.9  # 最优区域的 α
        alpha_min = 0.5  # 最差区域的 α

        scale_groups = defaultdict(list)
        for key, val in self.dist_mse_2_map.items():
            _, _, scale = key
            scale_groups[scale].append((key, val["mse"]))

        for scale, items in scale_groups.items():
            # 按 mse 从小到大排序（质量从好到差）
            sorted_items = sorted(items, key=lambda x: x[1])
            n = len(sorted_items)
            for i, (key, _) in enumerate(sorted_items):
                # 线性插值计算 alpha，i=0 -> alpha_max, i=n-1 -> alpha_min
                ratio = i / max(n - 1, 1)  # 防止除以0
                alpha = alpha_max - (alpha_max - alpha_min) * ratio
                self.dist_mse_2_map[key]["alpha"] = alpha

        # 第三阶段：根据 dist_mse 选择合适的尺度进行融合
        for num, (t_cutoff, scale) in enumerate(zip(timesteps, scales)):
            if step_index >= t_cutoff:
                continue

            for h in range(0, h_l, scale):
                for w in range(0, w_l, scale):
                    # 如果已经跳过这个位置，则跳过
                    if (h, w, scale) in skip_map and skip_map[(h, w, scale)]:
                        continue
                    alpha=self.dist_mse_2_map[(h, w, scale)]["alpha"]
                    # window = int(scale * cover_ritio)
                    window = int(scale * alpha)
                    patch,dist_mse,dist_mse_2 = self.getwindow_latent(latent,latent_small, h, h_l, h_s, w, w_l, w_s, window, scale)
                    # 否则执行融合操作
                    if (w // scale) % 2 == 0:
                        value[:, :, h:h+window, w:w+window] += patch
                        count[:, :, h:h+window, w:w+window] += 1
                    else:
                        h_shift = h + scale - window
                        value[:, :, h_shift:h_shift+window, w:w+window] += patch
                        count[:, :, h_shift:h_shift+window, w:w+window] += 1
        return torch.where(count > 0, value / count, value)

################
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def getwindow_latent(self,latent,latent_small,h,h_l,h_s,w,w_l,w_s,window,scale):
        h_ratio=(h/h_l)
        w_ratio=(w/w_l)
        h_small=int(h_s*h_ratio)
        w_small=int(w_s*w_ratio)

        h_start=h_small
        h_end=h_small+window
        w_start=w_small
        w_end=w_small+window

        h_start_ref=h_small
        w_start_ref=w_small
        h_end_ref = int(h_s * (h_ratio + scale / h_l))
        w_end_ref = int(w_s * (w_ratio + scale / w_l))

        if h_small+window > latent_small.shape[2]:
            h_start=latent_small.shape[2]-window
            h_end=latent_small.shape[2]
        if w_small+window > latent_small.shape[3]:
            w_start=latent_small.shape[3]-window
            w_end=latent_small.shape[3]
        window_latent = latent_small[:, :,h_start:h_end,w_start:w_end]

        # 获取原始 latent 的对应区域用于质量比较
        region = latent_small[:, :, h_start_ref:h_end_ref, w_start_ref:w_end_ref] 
        region_resized = F.interpolate(region, size=(window, window), mode='bilinear', align_corners=False)
        smoothed_region = F.avg_pool2d(region_resized, kernel_size=3, stride=1, padding=1)
        smoothed_patch = F.avg_pool2d(window_latent, kernel_size=3, stride=1, padding=1)
        dist_mse = F.mse_loss(smoothed_patch, smoothed_region).item()

        # 计算嵌入平滑度
        region_l = latent[:, :, h:h+window, w:w+window]
        smoothed_l = F.avg_pool2d(region_l, kernel_size=3, stride=1, padding=1)
        dist_mse_2 = F.mse_loss(region_l, smoothed_l).item()
        # print(dist_mse)
        return window_latent, dist_mse, dist_mse_2


if __name__ == '__main__':
    torch.cuda.set_device(0)

    # os.environ["CUDA_VISIBLE_DEVICES"] ="5"
    guidance_scale=8
    seed=1035
    H=2048
    W=2048
    small_height=512
    small_width=512
    upsample_st="nearest"
    # prompt='a photo of the dolomites'
    # prompt='A photo of the waterfall in the jungle.'
    # prompt='A photo of a bustling city skyline at dusk.'
    prompt='A photo of stormy ocean with massive waves crashing against cliffs.'
    
    scale=[64,32,16]
    timestep=[10,15,20]
    cover_ritio=0.75
    # outfile='./out/'+str(seed)+'/New_TFP_'+str(cover_ritio)+'_rotate'+str(scale)+'_'+str(timestep)+'/large.png'
    # smallfile='./out/'+str(seed)+'/New_TFP_'+str(cover_ritio)+'_rotate'+str(scale)+'_'+str(timestep)+'/small.png'

    outfile=f'./alpha4_out/large_{cover_ritio}_seed{seed}_64to3_check2.png'
    smallfile='./alpha4_out/small_512.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--prompt', type=str, default=prompt)
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--H', type=int, default=H)
    parser.add_argument('--W', type=int, default=W)

    parser.add_argument('--small_height', type=int,default=small_height)
    parser.add_argument('--small_width', type=int,default=small_width)
    parser.add_argument('--scale', type=list, default=scale)
    parser.add_argument('--upsample_st', type=str, default=upsample_st)
    parser.add_argument('--cover_ritio', type=float, default=cover_ritio)

    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--guidance_scale', type=float,default=guidance_scale)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--timestep', type=list,default=timestep)

    parser.add_argument('--outfile', type=str, default=outfile)
    parser.add_argument('--smallfile', type=str, default=smallfile)
    
    opt = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str)
    # parser.add_argument('--negative', type=str,default='')
    # parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
    #                     help="stable diffusion version")
    # parser.add_argument('--H', type=int)
    # parser.add_argument('--W', type=int)
    # parser.add_argument('--scale', nargs='+', type=int)
    # parser.add_argument('--upsample_st', type=str)
    # parser.add_argument('--seed', type=int)
    # parser.add_argument('--steps', type=int,default=50)
    # parser.add_argument('--outfile', type=str)
    # parser.add_argument('--weightoutfile', type=str)
    # parser.add_argument('--smallfile', type=str)
    # parser.add_argument('--guidance_scale', type=float)
    # parser.add_argument('--timestep', nargs='+', type=int)
    # parser.add_argument('--small_height', type=int)
    # parser.add_argument('--small_width', type=int)
    # parser.add_argument('--gpu', type=str)
    # opt = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = GCDiffusion(device, opt.sd_version)

    img,imgs_small = sd.text2panorama(prompts=opt.prompt, 
                                                negative_prompts=opt.negative, 
                                                height=opt.H, width=opt.W, 
                                                num_inference_steps=opt.steps,
                                                guidance_scale=opt.guidance_scale,
                                                timesteps=opt.timestep,
                                                scales=opt.scale,
                                                upsample_st=opt.upsample_st,
                                                small_height=opt.small_height,
                                                small_width=opt.small_width,
                                                cover_ritio=opt.cover_ritio)

    # save image
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    ensure_dir(opt.outfile)
    ensure_dir(opt.smallfile)
    img.save(opt.outfile)
    imgs_small.save(opt.smallfile)

