import os

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, ConsistencyDecoderVAE
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

from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance

def getwindow_latent(latent_small,h,h1,h2,w,w1,w2,window):
    h_ratio=(h/h1)
    w_ratio=(w/w1)
    h_small=int(h2*h_ratio)
    w_small=int(w2*w_ratio)

    h_start=h_small
    h_end=h_small+window
    w_start=w_small
    w_end=w_small+window
    if h_small+window > latent_small.shape[2]:
        h_start=latent_small.shape[2]-window
        h_end=latent_small.shape[2]
    if w_small+window > latent_small.shape[3]:
        w_start=latent_small.shape[3]-window
        w_end=latent_small.shape[3]
    window_latent = latent_small[:, :,h_start:h_end,w_start:w_end]
    return window_latent
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


def get_views(panorama_height, panorama_width, window_size=128, stride=16):
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
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '2.1-r':
            model_key = "/mnt/data/project_kyh/weight/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/checkpoint-150000"
        elif self.sd_version == 'sdxl':
            model_key = "/mnt/data/project_kyh/weight/stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.vae.enable_tiling()
        # self.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", added_cond_kwargs={}).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

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
    def decode_weight(self, latents):
        scale_factor = 8
        latents = F.interpolate(latents, size=(latents.size(2)*scale_factor, latents.size(3)*scale_factor), mode='bilinear', align_corners=False)
        latents = latents[:,0,:,:]
        min_val = latents.min()
        max_val = latents.max()
        normalized_tensor = (latents - min_val) / (max_val - min_val)

        weight_imgs=normalized_tensor
        return weight_imgs

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5,timesteps=20,scales=1,small_height=256,small_width=1024,cover_ritios=0.75):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # 1. Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        # 2. Define large-scale grid and get views
        h1=int(height//8)
        w1=int(width//8)
        h2=int(small_height//8)
        w2=int(small_width//8)

        latent = torch.randn((1, self.unet.in_channels, h1, w1), device=self.device)

        views,views_bidi,(h_max_num,w_max_num) = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)


        latent_small = torch.randn((1, self.unet.in_channels, small_height // 8, small_width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()
                for h_start, h_end, w_start, w_end in views:
                    # TODO we can support batches, and pass multiple views at once to the unet
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    # value_temp[h_num][w_num]=latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                value = torch.where(count > 0, value / count, value) 
                value = value*1
                count.fill_(1)

                #small map
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_small_model_input = torch.cat([latent_small] * 2)
                # predict the noise residual
                noise_small_pred = self.unet(latent_small_model_input, t, encoder_hidden_states=text_embeds)['sample']
                # perform guidance
                noise_small_pred_uncond, noise_small_pred_cond = noise_small_pred.chunk(2)
                noise_small_pred = noise_small_pred_uncond + guidance_scale * (noise_small_pred_cond - noise_small_pred_uncond)
                # compute the denoising step with the reference model
                latent_small = self.scheduler.step(noise_small_pred, t, latent_small)['prev_sample']

                ######################################################
                for num,(timestep,scale,cover_ritio) in enumerate(zip(timesteps,scales,cover_ritios)):
                    if i < timestep and num==0:#如果是最大尺度
                        window=int(scale*cover_ritio)

                        for h_num, h in enumerate(torch.arange(0,h1, scale)):
                            for w_num, w in enumerate(torch.arange(0, w1, scale)):
                                window_latent=getwindow_latent(latent_small,h,h1,h2,w,w1,w2,window)

                                if w_num%2==0:
                                        value[:, :, h:h+window, w:w+window] +=window_latent #上左
                                        count[:, :, h:h+window, w:w+window] += 1
                                else:
                                        value[:, :, h+scale-window:h+scale, w:w+window] +=window_latent #下左
                                        count[:, :, h+scale-window:h+scale, w:w+window] += 1

                    elif i < timestep and num!=0:#如果是不是最大尺度
                        scale_up=scale*2
                        window=int(scale*cover_ritio)
                        for h_num, h in enumerate(torch.arange(0,h1, scale_up)):
                            for w_num, w in enumerate(torch.arange(0, w1, scale_up)):
                                window_list=[]
                                window_list_index=0

                                for h_piece_num, h_piece in enumerate(torch.arange(0,scale_up, scale)):
                                    for w_piece_num, w_piece in enumerate(torch.arange(0, scale_up, scale)):
                                        window_list.append(getwindow_latent(latent_small,h+h_piece,h1,h2,w+w_piece,w1,w2,window))
                                for _ in range(num):
                                    last_element = window_list.pop()  # 弹出最后一个元素
                                    window_list.insert(0, last_element)  # 将其插入到列表的开头

                                for h_piece_num, h_piece in enumerate(torch.arange(0,scale_up, scale)):
                                    for w_piece_num, w_piece in enumerate(torch.arange(0, scale_up, scale)):
                                        if w_piece_num%2==0:
                                                value[:, :, h+h_piece:h+h_piece+window, w+w_piece:w+w_piece+window] +=window_list[window_list_index] #上左
                                                count[:, :, h+h_piece:h+h_piece+window, w+w_piece:w+w_piece+window] += 1
                                        else:
                                                value[:, :, h+h_piece+scale-window:h+h_piece+scale, w+w_piece:w+w_piece+window] +=window_list[window_list_index] #下左
                                                count[:, :, h+h_piece+scale-window:h+h_piece+scale, w+w_piece:w+w_piece+window] += 1
                                        window_list_index+=1

                latent = torch.where(count > 0, value / count, value) 

            # Img latents -> imgs
            imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
            img = T.ToPILImage()(imgs[0].cpu())
            imgs_small = self.decode_latents(latent_small)  
            img_small = T.ToPILImage()(imgs_small[0].cpu())


        return img,img_small
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ="4"
    num_device=1
    guidance_scale=8
    seed=6
    H=2048
    W=2048
    small_height=512
    small_width=512
    prompt='a photo of the dolomites'
    scale=[32,16,8]
    timestep=[10,15,20]
    cover_ritio=[0.6,0.6,0.6]
    

    outfile='/mnt/data/project_kyh/sdxl/out_seed'\
            +str(seed)+'_scale'+str(scale)+'_timestep'+str(timestep)+'_cover_ritio'+str(cover_ritio)+'.png'
    smallfile='/mnt/data/project_kyh/sdxl/out_seed'\
            +str(seed)+'_small.png'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_version', type=str, default='sdxl', choices=['1.5', '2.0','2.1','2.1-r','sdxl'],
                        help="stable diffusion version")
    parser.add_argument('--prompt', type=str, default=prompt)
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--H', type=int, default=H)
    parser.add_argument('--W', type=int, default=W)

    parser.add_argument('--small_height', type=int,default=small_height)
    parser.add_argument('--small_width', type=int,default=small_width)
    parser.add_argument('--scale', type=list, default=scale)
    parser.add_argument('--cover_ritio', type=list, default=cover_ritio)

    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--guidance_scale', type=float,default=guidance_scale)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--timestep', type=list,default=timestep)

    parser.add_argument('--outfile', type=str, default=outfile)
    parser.add_argument('--smallfile', type=str, default=smallfile)
    parser.add_argument('--gpu', type=int,default=num_device)

    opt = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str)
    # parser.add_argument('--negative', type=str,default='')
    # # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0','2.1'],
    # parser.add_argument('--sd_version', type=str, default='2.1-r', choices=['1.5', '2.0','2.1','2.1-r','sdxl'],
    #                     help="stable diffusion version")
    # parser.add_argument('--H', type=int)
    # parser.add_argument('--W', type=int)

    # parser.add_argument('--small_height', type=int)
    # parser.add_argument('--small_width', type=int)
    # parser.add_argument('--scale', nargs='+', type=int)
    # parser.add_argument('--timestep', nargs='+', type=int)
    # parser.add_argument('--cover_ritio', nargs='+', type=float)

    # parser.add_argument('--seed', type=int)
    # parser.add_argument('--guidance_scale', type=float)
    # parser.add_argument('--steps', type=int, default=50)

    # parser.add_argument('--outfile', type=str)
    # parser.add_argument('--smallfile', type=str)
    # parser.add_argument('--gpu', type=int)
    # opt = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu
# 
    seed_everything(opt.seed)

    torch.cuda.set_device(opt.gpu)

    device = torch.device('cuda')

    sd = GCDiffusion(device, opt.sd_version)

    img,imgs_small = sd.text2panorama(prompts=opt.prompt, 
                                                negative_prompts=opt.negative, 
                                                height=opt.H, width=opt.W, 
                                                num_inference_steps=opt.steps,
                                                guidance_scale=opt.guidance_scale,
                                                timesteps=opt.timestep,
                                                scales=opt.scale,
                                                small_height=opt.small_height,
                                                small_width=opt.small_width,
                                                cover_ritios=opt.cover_ritio)

    # save image
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    ensure_dir(opt.outfile)
    ensure_dir(opt.smallfile)
    img.save(opt.outfile)
    imgs_small.save(opt.smallfile)

