"""
Util functions based on Diffuser framework.
"""


import os
import torch
import cv2
import numpy as np
import torch.nn.functional as nnf

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from typing import Optional, Union, Tuple, List, Callable, Dict
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from torch.optim.adam import Adam
from torchvision.io import read_image

from diffusers import StableDiffusionPipeline

from pytorch_lightning import seed_everything


NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
 #New
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

class MasaCtrlPipeline(StableDiffusionPipeline):

    # def next_step(
    #     self,
    #     model_output: torch.FloatTensor,
    #     timestep: int,
    #     x: torch.FloatTensor,
    #     eta=0.,
    #     verbose=False
    # ):
    #     """
    #     Inverse sampling for DDIM Inversion
    #     """
    #     if verbose:
    #         print("timestep: ", timestep)
    #     next_step = timestep
    #     timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
    #     beta_prod_t = 1 - alpha_prod_t
    #     pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    #     pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
    #     x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
    #     return x_next, pred_x0


    # New
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    # New
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    # New
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context).sample
        return noise_pred

    # New
    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context).sample
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    #New
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image
    
    #New
    @torch.no_grad()
    def image2latent(self, image):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    #New
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    # New
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    #New
    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    # New
    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                # if j == (num_inner_steps-1):
                #     print(f'Step == {i}, Iteration == {j}, Loss == {loss}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
   
    # New
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        self.scheduler.set_timesteps(50) # num_inference_steps
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    # @torch.no_grad()
    # def image2latent(self, image):
    #     DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     if type(image) is Image:
    #         image = np.array(image)
    #         image = torch.from_numpy(image).float() / 127.5 - 1
    #         image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    #     # input image density range [-1, 1]
    #     latents = self.vae.encode(image)['latent_dist'].mean
    #     latents = latents * 0.18215
    #     return latents

    # @torch.no_grad()
    # def latent2image(self, latents, return_type='np'):
    #     latents = 1 / 0.18215 * latents.detach()
    #     image = self.vae.decode(latents)['sample']
    #     if return_type == 'np':
    #         image = (image / 2 + 0.5).clamp(0, 1)
    #         image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    #         image = (image * 255).astype(np.uint8)
    #     elif return_type == "pt":
    #         image = (image / 2 + 0.5).clamp(0, 1)

    #     return image

    # def latent2image_grad(self, latents):
    #     latents = 1 / 0.18215 * latents
    #     image = self.vae.decode(latents)['sample']

    #     return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        temp = None,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if isinstance(guidance_scale, list):
            guid_scale = guidance_scale[0]
        else:
            guid_scale = guidance_scale
        print(guid_scale)
        if guid_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guid_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                if temp is not None:
                    text_embeddings = torch.cat([torch.cat([unconditioning[i],temp[i],temp[i]]), text_embeddings])
                else:
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guid_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                if isinstance(guidance_scale, list):
                    all = []
                    for i in range(len(guidance_scale)):
                        all.append(noise_pred_uncon[i] + guidance_scale[i] * (noise_pred_con[i] - noise_pred_uncon[i]))
                    noise_pred = torch.stack(all, dim=0)
                else:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    # @torch.no_grad()
    # def invert(
    #     self,
    #     image: torch.Tensor,
    #     prompt,
    #     num_inference_steps=50,
    #     guidance_scale=7.5,
    #     eta=0.0,
    #     return_intermediates=False,
    #     **kwds):
    #     """
    #     invert a real image into noise map with determinisc DDIM inversion
    #     """
    #     DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     batch_size = image.shape[0]
    #     if isinstance(prompt, list):
    #         if batch_size == 1:
    #             image = image.expand(len(prompt), -1, -1, -1)
    #     elif isinstance(prompt, str):
    #         if batch_size > 1:
    #             prompt = [prompt] * batch_size

    #     # text embeddings
    #     text_input = self.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=77,
    #         return_tensors="pt"
    #     )
    #     text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
    #     print("input text embeddings :", text_embeddings.shape)
    #     # define initial latents
    #     latents = self.image2latent(image)
    #     start_latents = latents
    #     # print(latents)
    #     # exit()
    #     # unconditional embedding for classifier free guidance
    #     if guidance_scale > 1.:
    #         max_length = text_input.input_ids.shape[-1]
    #         unconditional_input = self.tokenizer(
    #             [""] * batch_size,
    #             padding="max_length",
    #             max_length=77,
    #             return_tensors="pt"
    #         )
    #         unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
    #         text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

    #     # self.unconditional_embeddings = unconditional_embeddings
    #     # self.conditional_embeddings = text_embeddings
    #     self.context = text_embeddings

    #     print("latents shape: ", latents.shape)
    #     # interative sampling
    #     self.scheduler.set_timesteps(num_inference_steps)
    #     print("Valid timesteps: ", reversed(self.scheduler.timesteps))
    #     # print("attributes: ", self.scheduler.__dict__)
    #     latents_list = [latents]
    #     pred_x0_list = [latents]
    #     for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
    #         if guidance_scale > 1.:
    #             model_inputs = torch.cat([latents] * 2)
    #         else:
    #             model_inputs = latents

    #         # predict the noise
    #         noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
    #         if guidance_scale > 1.:
    #             noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
    #             noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
    #         # compute the previous noise sample x_t-1 -> x_t
    #         latents, pred_x0 = self.next_step(noise_pred, t, latents)
    #         latents_list.append(latents)
    #         pred_x0_list.append(pred_x0)

    #     if return_intermediates:
    #         # return the intermediate laters during inversion
    #         # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
    #         return latents, latents_list
    #     return latents, start_latents
    

    