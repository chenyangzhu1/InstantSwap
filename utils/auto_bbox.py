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

    def __init__(self, cross_attn_map_store, self_attn_map_store) -> None:
        self.cross_attn_map_store = cross_attn_map_store
        self.self_attn_map_store = self_attn_map_store

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        # print(encoder_hidden_states)
        if encoder_hidden_states is not None:  # cross attn
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
            self.cross_attn_map_store.append(attention_probs)
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
            self.self_attn_map_store.append(attention_probs)
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
