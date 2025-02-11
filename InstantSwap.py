from typing import Tuple, Union, Optional, List
import os

import argparse
import torchvision.transforms as transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--model_id", type=str, help="Model ID")
    parser.add_argument("--source_mask", type=str, help="Source Mask")
    parser.add_argument("--source_image", type=str, help="Source Image")
    parser.add_argument("--source_prompt", type=str, help="Source Prompt")
    parser.add_argument("--target_prompt", type=str, help="Target Prompt")
    parser.add_argument("--output", type=str, help="Output")
    parser.add_argument("--iters", type=int, help="iters")
    parser.add_argument("--diff_prompt", type=str, help="diff_prompt")
    parser.add_argument("--diff_prompt_source", type=str)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--interval", type=int)

    return parser.parse_args()


def save_args(argsDict, root):
    with open(root, "w") as f:
        f.writelines("------------------ start ------------------" + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
        f.writelines("------------------- end -------------------")


args = parse_arguments()
args.output = (
    args.output
    + str(args.iters)
    + "_guidance_"
    + str(args.guidance_scale)
    + "_interval_"
    + str(args.interval)
    + "/"
)
if not os.path.exists(args.output):
    print("create dir!")
    os.makedirs(args.output)
arg_root = os.path.join(args.output, "setting.txt")
save_args(args.__dict__, arg_root)
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
from utils.Attn_proce import MyAttnProcessor

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
        region_mask=None,
        embedding_region=None,
        height=None,
        width=None,
        tag=0,
    ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(
            -1, *text_embeddings.shape[2:]
        )
        if embedding_region is not None:
            embedd_region = embedding_region.permute(1, 0, 2, 3).reshape(
                -1, *embedding_region.shape[2:]
            )
        else:
            embedd_region = embedding_region
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                latent_input,
                timestep,
                embedd,
                cross_attention_kwargs={
                    "region1_hidden": embedd_region,
                    "region1_bbox": region_mask,
                    "height": height,
                    "width": width,
                },
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

    def get_bgm_loss(
        self,
        z_source: T,
        z_target: T,
        text_emb_source: T,
        text_emb_target: T,
        eps=None,
        reduction="mean",
        symmetric: bool = False,
        timestep: Optional[int] = None,
        guidance_scale=7.5,
        mask_source_out=None,
        region_mask=None,
        embedding_region=None,
        embedding_region_source=None,
        height=None,
        width=None,
        iter=None,
        interval=None,
    ) -> TS:
        with torch.inference_mode():
            if iter % interval == 0:
                z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(
                    z_source, eps, timestep
                )
                z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
                eps_pred_source, _ = self.get_eps_prediction(
                    z_t_source,
                    timestep,
                    text_emb_source,
                    alpha_t,
                    sigma_t,
                    guidance_scale=guidance_scale,
                    region_mask=region_mask,
                    embedding_region=embedding_region_source,
                    height=height,
                    width=width,
                    tag=0,
                )
                eps_pred_target, _ = self.get_eps_prediction(
                    z_t_target,
                    timestep,
                    text_emb_target,
                    alpha_t,
                    sigma_t,
                    guidance_scale=guidance_scale,
                    region_mask=region_mask,
                    embedding_region=embedding_region,
                    height=height,
                    width=width,
                    tag=1,
                )

                """
                mask process begin
                """
                mask_source_out[mask_source_out > 0] = 1
                res = int((eps_pred_source.shape[3]))  # 64
                mask_fuse = F.interpolate(
                    mask_source_out.unsqueeze(0).unsqueeze(0), (res, res)
                )
                """
                mask process end
                """
                grad = (
                    (alpha_t**self.alpha_exp)
                    * (sigma_t**self.sigma_exp)
                    * (eps_pred_target - eps_pred_source)
                )
                grad = grad * mask_fuse
                grad = grad
                self.former_grad = grad.clone()  # for IGS
            else:
                grad = self.former_grad.clone()
        loss = z_target * grad.clone()
        if symmetric:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == "mean":
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss

    def get_grad(self):
        return self.former_grad

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


model_id = args.model_id
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)


def get_attn_str(attn_name):
    index = attn_name.rfind(".")
    substring = attn_name[:index]
    attn1_str = substring.split(".")[-1]
    return attn1_str


attn_processor_dict = {}
for k in pipeline.unet.attn_processors.keys():
    if get_attn_str(k) == "attn2":
        attn_processor_dict[k] = MyAttnProcessor()
    else:
        attn_processor_dict[k] = MyAttnProcessor()

pipeline.unet.set_attn_processor(attn_processor_dict)


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


def image_optimization(
    pipe: StableDiffusionPipeline,
    image: np.ndarray,
    text_source: str,
    text_target: str,
    mask_source_out: str,
    num_iters=200,
    region_mask=None,
    embedding_region=None,
    embedding_region_source=None,
    height=None,
    width=None,
    output=None,
    guidance_scale=7.5,
    interval=None,
) -> None:
    bgm_loss = BGMLoss(device, pipe)
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)["latent_dist"].mean * 0.18215
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text = get_text_embeddings(pipeline, text_source)
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    image_target.requires_grad = True
    z_taregt = z_source.clone()
    z_taregt.requires_grad = True
    optimizer = SGD(params=[z_taregt], lr=1e-1)
    timestep_list = []
    for i in range(num_iters):
        timestep = torch.randint(
            low=100,
            high=900,  # Avoid the highest timestep.
            size=(z_taregt.shape[0],),
            device=z_taregt.device,
            dtype=torch.long,
        )
        timestep_list.append(timestep)
    timestep_list.sort(reverse=True)
    for i in tqdm(range(num_iters)):
        loss = bgm_loss.get_bgm_loss(
            z_source,
            z_taregt,
            embedding_source,
            embedding_target,
            mask_source_out=mask_source_out,
            timestep=timestep_list[i],
            region_mask=region_mask,
            embedding_region=embedding_region,
            embedding_region_source=embedding_region_source,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            iter=i,
            interval=interval,
        )
        optimizer.zero_grad()
        (2000 * loss).backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            out = decode(z_taregt, pipeline, im_cat=None)
            out.save(
                output
                + text_source.replace(" ", "_")
                + "-->"
                + text_target.replace(" ", "_")
                + str(i)
                + ".png"
            )
        if i == num_iters - 1:
            out = decode(z_taregt, pipeline, im_cat=None)
            out.save(output + "final.png")


transform = transforms.Compose([transforms.ToTensor()])
image = Image.open(args.source_mask)
image_tensor = transform(image).to(device)
mask_source_out = image_tensor.mean(dim=0)  # torch.Size([512, 512])


def mask_find_bboxs(mask):
    non_zero_indices = torch.nonzero(mask > 0)
    top_left = (
        non_zero_indices[:, 0].min().item() / mask.shape[0],
        non_zero_indices[:, 1].min().item() / mask.shape[0],
    )
    bottom_right = (
        non_zero_indices[:, 0].max().item() / mask.shape[0],
        non_zero_indices[:, 1].max().item() / mask.shape[0],
    )
    return top_left[0], bottom_right[0], top_left[1], bottom_right[1]


region_mask = mask_find_bboxs(mask_source_out)
embedding_region = torch.stack(
    [
        get_text_embeddings(pipeline, ""),
        get_text_embeddings(pipeline, args.diff_prompt),
    ],
    dim=1,
)
embedding_region_source = torch.stack(
    [
        get_text_embeddings(pipeline, ""),
        get_text_embeddings(pipeline, args.diff_prompt_source),
    ],
    dim=1,
)

image = load_512(args.source_image)
image_optimization(
    pipeline,
    image,
    args.source_prompt,
    args.target_prompt,
    mask_source_out=mask_source_out,
    num_iters=args.iters,
    output=args.output,
    region_mask=region_mask,
    embedding_region=embedding_region,
    embedding_region_source=embedding_region_source,
    height=512,
    width=512,
    guidance_scale=args.guidance_scale,
    interval=args.interval,
)
