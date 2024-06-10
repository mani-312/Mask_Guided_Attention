import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
import cv2
import ptp_utils
import argparse

import numpy as np
from PIL import Image

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline, load_512
from masactrl.masactrl_utils import AttentionBase, AttentionStore
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

torch.cuda.set_device(1)  # set the GPU device



# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model_path = "xyn-ai/anything-v4.0"
model_path = 'CompVis/stable-diffusion-v1-4'
# model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5




@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    # ptp_utils.register_attention_control(model, controller)
    if controller is not None:
        regiter_attention_editor_diffusers(model,controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller = None, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    images, x_t = text2image_ldm_stable(model, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t




seed = 100
seed_everything(seed)


# examples_appearance = [
#     [
#         "examples/appearance/001_base.png",
#         "examples/appearance/001_replace.png",
#         'a photo of a cake',
#         'a photo of a cake',
#     ],
#     [
#         "examples/appearance/002_base.png",
#         "examples/appearance/002_replace.png",
#         'a photo of a doughnut',
#         'a photo of a doughnut',
#     ],
#     [
#         "examples/appearance/003_base.jpg",
#         "examples/appearance/003_replace.png",
#         'a photo of a Swiss roll',
#         'a photo of a Swiss roll',
#     ],
#     [
#         "examples/appearance/004_base.jpg",
#         "examples/appearance/004_replace.jpeg",
#         'a photo of a car',
#         'a photo of a car',
#     ],
#     [
#         "examples/appearance/005_base.jpeg",
#         "examples/appearance/005_replace.jpg",
#         'a photo of an ice-cream',
#         'a photo of an ice-cream',
#     ],
# ]

# Sofa
# person_prompt = "A photo of a sofa in a room"
# garment_prompt = "A photo of a sofa"


# person_prompt = "A photo of a suitcase in the beach"
# garment_prompt = "A photo of a suitcase in the beach"


# "A photo of a giraffe"
# "A photo of a zebra"
# 'A photo of a tiger'


# "A photo of a white horse in the garden"
# "A photo of a red horse in the garden"

# 'A photo of a duomo'
# 'A photo of a taj mahal'

# 'A photo of a dog'



parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('-s', '--src', type=str, help='Source file name') # Appearance image
parser.add_argument('-t', '--target', type=str, help='Target file name') # Structure image
parser.add_argument('-t_p', '--target_prompt', type=str, help='Target prompt name')
parser.add_argument('-s_p', '--src_prompt', type=str, help='Source prompt name')

args = parser.parse_args()


person_path = args.target
person_prompt = args.target_prompt

garment_path = args.src
garment_prompt = args.src_prompt

src_base = args.src.split('.')[0]
target_base = args.target.split('.')[0]
src_mask_path = src_base + '_mask.pth'
target_mask_path = target_base + '_mask.pth'
src_nti_path = src_base + '_nti.pth'
target_nti_path = target_base + '_nti.pth'

print(src_base)
print(target_base)


## Null-Text Inversion
if not os.path.exists(target_nti_path):
    (image_gt, image_enc), person_x_t, person_uncond_embeddings = model.invert(person_path, person_prompt, num_inner_steps = 10, offsets=(0,0,0,0), verbose=True)

    torch.save({'x_t': person_x_t, 'uncond_embeddings': person_uncond_embeddings}, target_nti_path)


saved_vars = torch.load(target_nti_path)

# Retrieve x_t and uncond_embeddings
person_x_t = saved_vars['x_t'].to(device)
person_uncond_embeddings = [i.to(device) for i in saved_vars['uncond_embeddings']]


if not os.path.exists(src_nti_path):
    (image_gt, image_enc), garment_x_t, garment_uncond_embeddings = model.invert(garment_path, garment_prompt, num_inner_steps = 10, offsets=(0,0,0,0), verbose=True)

    torch.save({'x_t': garment_x_t, 'uncond_embeddings': garment_uncond_embeddings}, src_nti_path)


saved_vars = torch.load(src_nti_path)

# Retrieve x_t and uncond_embeddings
garment_x_t = saved_vars['x_t'].to(device)
garment_uncond_embeddings = [i.to(device) for i in saved_vars['uncond_embeddings']]


def pt_to_np(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


mask_s = torch.from_numpy(torch.load(src_mask_path)).float()
kernel = np.ones((3, 3), dtype=np.uint8)
mask_t = torch.from_numpy(cv2.dilate(torch.load(target_mask_path).astype('uint8'), kernel, iterations=5)).float()


from masactrl.masactrl import MutualSelfAttentionControlMask_ComposeSyntheisized, MutualSelfAttentionControlMask_ComposeReference, MutualSelfAttentionControlMask_AT_StructureBackground, MutualSelfAttentionControlMask_AT_AppearanceBackground
from torchvision.io import read_image


# inference the synthesized image with MasaCtrl
# STEP = 4
STEP = 4
LAYPER = 10

mask_s = mask_s.to(device)
mask_t = mask_t.to(device)

# Output image background coming from Structure image(target_ne)
editor = MutualSelfAttentionControlMask_AT_StructureBackground(start_step = STEP,start_layer = LAYPER, mask_s = mask_s, mask_t = mask_t)

# Output image background coming from Appearance image(src)
# editor = MutualSelfAttentionControlMask_AT_AppearanceBackground(start_step = STEP,start_layer = LAYPER, mask_s = mask_s, mask_t = mask_t)

# Composing a synthesized new object
# src_image => corgi
# src_prompt = A photo of a corgi
# target_prompt = A photo of a person holding corgi (person is new object)
# mask_s = mask of corgi in real image
# mask_t = mask of corgi+person in synthesized image with target prompt
# mask_new_object = mask of person in synthesized image                                                                                     
# editor = MutualSelfAttentionControlMask_ComposeSyntheisized(start_step = STEP,start_layer = LAYPER, mask_s = mask_s, mask_t = mask_t,mask_new_object=mask_new)
                                                                                                                                           

# Composing a new reference object
# src_image => corgi
# ref_image => person
# src_prompt = A photo of a corgi
# ref_prompt = A photo of a person
# target_prompt = A photo of a person holding corgi (person is new object)
# mask_s = mask of corgi in real image
# mask_t_s = mask of corgi in synthesized image with target prompt
# mask_t_ref = mask of person in synthesized image with target prompt
# mask_ref = mask of person in ref image                                                                                                                                            
# editor = MutualSelfAttentionControlMask_ComposeReference(start_step = STEP,start_layer = LAYPER, mask_s = mask_s,mask_t_s=mask_t_s, mask_t_ref = mask_t_ref, mask_ref = mask_ref)
regiter_attention_editor_diffusers(model, editor)

prompts = [garment_prompt, person_prompt, person_prompt]
start_code = torch.cat([garment_x_t,person_x_t, person_x_t])
image_masactrl = model(prompts,
                       latents=start_code,
                       guidance_scale=7.5,
                       unconditioning = garment_uncond_embeddings,
                       temp = person_uncond_embeddings)

image_garment = load_512(garment_path)
image_person = load_512(person_path)
final_images = pt_to_np(image_masactrl)

print(image_person.shape)
print(image_masactrl.shape)

# transfer object in person_prompt to image_masactrl[1] with mask.
# kernel = np.ones((3, 3), dtype=np.uint8)
# mask = 1.0 - torch.from_numpy(cv2.dilate(torch.load(target_mask_path).astype('uint8'), kernel, iterations=1))
# final_images[1][mask==1] = torch.from_numpy(image_person)[mask==1]
# final_images[2][mask==1] = torch.from_numpy(image_person)[mask==1]


folder = target_base.split('/')[-1]
out_dir = f"./workdir/masactrl_real_exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir)) + 1
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

# Appearance image, Reconstructed appearance image, Structure image, Reconstructed structure image, Output image
pil_image = ptp_utils.view_images([image_garment,final_images[0], image_person, final_images[1], final_images[2]])

pil_image.save(os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
pil_image = ptp_utils.view_images([final_images[2]])
pil_image.save(os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}_result.png"))