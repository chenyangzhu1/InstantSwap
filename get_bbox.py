from typing import Tuple, Union, Optional, List
import os
from scipy import ndimage
import numpy as np


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--model_id", type=str, help="Model ID")
    parser.add_argument("--source_image", type=str, help="Source Image")
    parser.add_argument("--source_prompt", type=str, help="Source Prompt")
    parser.add_argument("--output", type=str, help="Output")
    parser.add_argument("--iters", type=int, help="iters")
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--word_idx", type=int)

    return parser.parse_args()


args = parse_arguments()
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython.display import display, clear_output
import torch.nn.functional as F
from utils.auto_bbox import MyAttnProcessor

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

device = torch.device("cuda:0")

import random


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed = 42
seed_everything(seed)


def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer(
        [text],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline, im_cat: TN = None):
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)


def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class BGMLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]

            timestep = torch.randint(
                low=100,
                high=900,  # Avoid the highest timestep.
                size=(b,),
                device=z.device,
                dtype=torch.long,
            )
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(
        self,
        z_t: T,
        timestep: T,
        text_embeddings: T,
        alpha_t: T,
        sigma_t: T,
        get_raw=False,
        guidance_scale=7.5,
    ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(
            -1, *text_embeddings.shape[2:]
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                latent_input,
                timestep,
                embedd,
            ).sample
            if self.prediction_type == "v_prediction":
                e_t = (
                    torch.cat([alpha_t] * 2) * e_t
                    + torch.cat([sigma_t] * 2) * latent_input
                )
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t

            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)  # cfg
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def clear_list(self):
        self.cross_attn_map_store.clear()
        self.self_attn_map_store.clear()

    def get_bgm_loss(
        self,
        z_source: T,
        text_emb_source: T,
        eps=None,
        timestep: Optional[int] = None,
        guidance_scale=7.5,
        height=None,
        width=None,
        iter=None,
        output=None,
    ) -> TS:
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(
                z_source, eps, timestep
            )
            eps_pred_source, _ = self.get_eps_prediction(
                z_t_source,
                timestep,
                text_emb_source,
                alpha_t,
                sigma_t,
                guidance_scale=guidance_scale,
            )
        return self.cross_attn_map_store, self.self_attn_map_store

    def get_grad(self):
        return self.former_grad

    def get_attn_str(self, attn_name):
        index = attn_name.rfind(".")
        substring = attn_name[:index]
        attn1_str = substring.split(".")[-1]
        return attn1_str

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(
            device, dtype, pipe.unet, pipe.scheduler
        )
        self.prediction_type = pipe.scheduler.prediction_type
        self.former_grad = None
        self.cross_attn_map_store = []
        self.self_attn_map_store = []
        attn_processor_dict = {}
        for k in pipe.unet.attn_processors.keys():
            if self.get_attn_str(k) == "attn2":
                attn_processor_dict[k] = MyAttnProcessor(
                    self.cross_attn_map_store, self.self_attn_map_store
                )
            else:
                attn_processor_dict[k] = MyAttnProcessor(
                    self.cross_attn_map_store, self.self_attn_map_store
                )

        pipe.unet.set_attn_processor(attn_processor_dict)


model_id = args.model_id
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

import os


def sample_and_sort(t_min, t_max, num_iters, z_taregt):
    samples = torch.linspace(
        t_min,
        t_max,
        num_iters,
        device=z_taregt.device,
        dtype=torch.long,
    )
    sorted_samples, _ = torch.sort(samples, descending=True)
    ret_list = [tensor.view(1) for tensor in sorted_samples]
    return ret_list


def masks_to_boxes(mask: np.array) -> np.array:

    x, y = np.where(mask != 0)

    st_h = np.min(x)
    st_w = np.min(y)
    ed_h = np.max(x)
    ed_w = np.max(y)

    return st_h, st_w, ed_h, ed_w


def image_optimization(
    pipe: StableDiffusionPipeline,
    image: np.ndarray,
    text_source: str,
    num_iters=200,
    height=None,
    width=None,
    output=None,
    guidance_scale=7.5,
    word_idx=None,
) -> None:
    rds_loss = BGMLoss(device, pipe)
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)["latent_dist"].mean * 0.18215
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text = get_text_embeddings(pipeline, text_source)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)

    image_target.requires_grad = True

    z_taregt = z_source.clone()
    z_taregt.requires_grad = True

    timestep_list = sample_and_sort(100, 500, num_iters, z_taregt)

    file_pairs = []
    for i in tqdm(range(num_iters)):
        cross_data, self_data = rds_loss.get_bgm_loss(
            z_source,
            embedding_source,
            timestep=timestep_list[i],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            iter=i,
            output=output,
        )
    for _ in range(num_iters):
        cross_attn = []
        self_attn = []
        for i in range(16):
            cross_attn.append(rds_loss.cross_attn_map_store.pop(0))
            self_attn.append(rds_loss.self_attn_map_store.pop(0))

        file_pairs.append((cross_attn, self_attn))

    masks = []
    for cross_attn, self_attn in file_pairs:
        cross_256 = []
        self_256 = []
        for tensor in cross_attn:
            if tensor.shape[1] == 256:
                cross_256.append(tensor.reshape(-1, 16 * 16, 77))

        for tensor in self_attn:
            if tensor.shape[1] == 256:
                self_256.append(tensor.reshape(-1, 16 * 16, 16 * 16))

        cross_256 = torch.cat(cross_256, dim=0)
        attention_maps = cross_256.sum(dim=0) / cross_256.shape[0]
        attention_maps = torch.pow(attention_maps, 2)
        self_256 = torch.cat(self_256, dim=0)
        self_maps = self_256.sum(dim=0) / self_256.shape[0]
        attention_maps = (self_maps @ attention_maps).reshape(16, 16, 77).cpu()
        image = attention_maps[:, :, word_idx]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((512, 512)))
        image_normalized = image / 255.0
        image_normalized = image_normalized.sum(axis=2) / 3
        image_normalized[image_normalized >= 0.5] = 1
        image_normalized[image_normalized < 0.5] = 0
        labels, num_features = ndimage.label(image_normalized)
        sizes = ndimage.sum(image_normalized, labels, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        image_cleaned = np.copy(image_normalized)
        image_cleaned[labels != largest_label] = 0
        masks.append(image_cleaned)
    final_mask = np.zeros((512, 512))

    for mask in masks:
        final_mask = np.logical_or(final_mask, mask)

    st_h, st_w, ed_h, ed_w = masks_to_boxes(final_mask)
    bbox_out = np.zeros([512, 512])
    bbox_out[st_h:ed_h, st_w:ed_w] = 1
    bbox_cross = Image.fromarray((bbox_out * 255.0).astype(np.uint8))
    bbox_cross.save(os.path.join(output, "bbox.jpg"))


image = load_512(args.source_image)

image_optimization(
    pipeline,
    image,
    args.source_prompt,
    num_iters=args.iters,
    output=args.output,
    height=512,
    width=512,
    guidance_scale=args.guidance_scale,
    word_idx=args.word_idx,
)
