from diffusers.models.attention_processor import Attention
import torch
from typing import Callable, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
import random
from einops import rearrange
import math
from torch import einsum


class MyAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        region1_hidden: Optional[torch.FloatTensor] = None,
        region1_bbox: Optional[tuple] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        if (
            encoder_hidden_states is not None
            and region1_hidden is not None
            and region1_bbox is not None
        ): # cross attn
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
            key_region1 = attn.to_k(region1_hidden, *args)
            value_region1 = attn.to_v(region1_hidden, *args)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            """
            使用self Attention 传来的Hidden states 
            只修改region部分
            torch.Size([2, 4096, 320])  torch.Size([10, 4096, 64])
            """
            # hidden_states = torch.bmm(attention_probs, value) 
            hidden_states=attn.head_to_batch_dim(hidden_states)

            seq_lens = query.shape[1] # 9216  4096
            downscale = math.sqrt(height * width / seq_lens)# 8
            height, width = int(height // downscale), int(width // downscale)

            """
            mask codes
            """
            mask4region1 = torch.zeros((height, width))
            st_h1_r, ed_h1_r, st_w1_r, ed_w1_r = region1_bbox
            start_h1 = int(st_h1_r * height)
            end_h1 = int(ed_h1_r * height)
            start_w1 = int(st_w1_r * width)
            end_w1 = int(ed_w1_r * width)
            mask4region1[start_h1:end_h1, start_w1:end_w1] += 1

            region_mask = mask4region1

            my_query = rearrange(query, "b (h w) c -> b h w c", h=height, w=width)
            hidden_states = rearrange(
                hidden_states, "b (h w) c -> b h w c", h=height, w=width
            )

            new_hidden_state = torch.zeros_like(hidden_states)
            new_hidden_state[:, region_mask == 0, :] = hidden_states[
                :, region_mask == 0, :
            ]
            replace_ratio = 1.0
            new_hidden_state[:, region_mask != 0, :] = (
                1 - replace_ratio
            ) * hidden_states[:, region_mask != 0, :]

            key_region1 = attn.head_to_batch_dim(key_region1)

            value_region1 = attn.head_to_batch_dim(value_region1)

            attention_region_retion1 = (
                einsum(
                    "b h w c, b n c -> b h w n",
                    my_query[:, start_h1:end_h1, start_w1:end_w1, :],
                    key_region1,
                )
                * attn.scale
            )

            if attn.upcast_softmax:
                attention_region_retion1 = attention_region_retion1.float()

            attention_region_retion1 = attention_region_retion1.softmax(dim=-1)

            attention_region_retion1 = attention_region_retion1.to(query.dtype)

            hidden_state_region_region1 = einsum(
                "b h w n, b n c -> b h w c", attention_region_retion1, value_region1
            )
            replace_ratio = 1
            new_hidden_state[
                :, start_h1:end_h1, start_w1:end_w1, :
            ] += replace_ratio * (
                hidden_state_region_region1
                / (
                    region_mask.reshape(1, *region_mask.shape, 1)[
                        :, start_h1:end_h1, start_w1:end_w1, :
                    ]
                ).to(query.device)
            )

            new_hidden_state = rearrange(new_hidden_state, "b h w c -> b (h w) c")

            hidden_states = attn.batch_to_head_dim(new_hidden_state)
            # hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        else:
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
