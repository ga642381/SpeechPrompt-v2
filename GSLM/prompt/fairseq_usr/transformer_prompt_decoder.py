# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig, TransformerDecoderBase
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase

from torch import Tensor
from torch.nn import Dropout, ModuleList


class TransformerDecoderPromptLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, return_fc=False):
        super().__init__(cfg, return_fc)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        key_prompt=None,
        value_prompt=None,
        prompt_type=None,
        x_sep=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            # ==== Prefix Tuning ==== #
            # https://arxiv.org/abs/2101.00190
            # x shape [81, 32, 1024]
            if key_prompt is not None and value_prompt is not None and prompt_type is not None:
                key_y = torch.clone(x)
                value_y = torch.clone(x)

                # prefix
                if prompt_type == "prefix":
                    key_prompt = key_prompt.unsqueeze(1).repeat(1, x.shape[1], 1)  # [10, 1, 1024]
                    value_prompt = value_prompt.unsqueeze(1).repeat(1, x.shape[1], 1)  # [10, 1, 1024]
                    key_y[: key_prompt.shape[0]] = key_prompt
                    value_y[: value_prompt.shape[0]] = value_prompt

                # infix
                elif prompt_type == "infix" and x_sep is not None:
                    prompt_length = key_prompt.shape[0]
                    for (i, sep_i) in x_sep:
                        key_y[sep_i : sep_i + prompt_length, i] = key_prompt
                        value_y[sep_i : sep_i + prompt_length, i] = value_prompt

                else:
                    raise NotImplementedError("Not implemented!")

            else:
                # no prompt
                key_y = x
                value_y = x

        x, attn = self.self_attn(
            query=x,
            key=key_y,
            value=value_y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


class TransformerDecoderPromptBase(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        prompts,
        embed_sep_token=None,
        verbalizer=None,
        no_encoder_attn=True,
        output_projection=None,
    ):
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn,
            output_projection,
        )
        # we cannot use args.use_sep_token because we have already pass into the TransformerConfig in TransformerDecoder, but we actually don't need it
        self.use_sep_token = True if embed_sep_token is not None else False
        self.sep = 0
        self.prefix_embed_prompts = prompts.dec_input_prompt
        self.infix_embed_prompts = None
        self.sep_embed = embed_sep_token
        self.deep_prompt = prompts.deep_prompt
        self.deep_key_embed_prompts = prompts.dec_key_prompt
        self.deep_value_embed_prompts = prompts.dec_value_prompt
        self.prefix_prompt_length = prompts.prompt_length
        self.infix_prompt_length = 0
        self.prompt_dropout = Dropout(p=0.1)
        self.fine_tune = False

        ###### Linear Verbalizer ######
        self.linear_verbalizer = True if verbalizer is not None else False
        self.prompt_verbalizer_dropout = Dropout(p=0.1)
        if self.linear_verbalizer:
            self.prompt_linear_verbalizer = verbalizer.linear_verbalizer
            self.num_classes = verbalizer.num_classes

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
        )

        # x.shape : [32, 1, 1024]
        if not features_only:
            x = self.output_layer(x)
        # x.shape : [32, 1. num_classes]
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            src_lengths,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # [Prompt] not using incremental decoding except when fine-tuning
        if not self.fine_tune:
            incremental_state = None

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # ===== <embed positions> ===== #
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)

        token_embed = self.embed_tokens(prev_output_tokens)

        # ======= <sep> ======= #
        sep_embed = self.sep_embed(torch.tensor(self.sep).cuda())
        # replace sep with trainable sep token
        if self.use_sep_token:
            sep_position = (prev_output_tokens == 0).nonzero()
            for (i, p) in sep_position:
                token_embed[int(i)][int(p)] = sep_embed
        # ======= </sep> ======= #
        x = self.embed_scale * token_embed

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # save input embedding
        embed = x

        # ===== </embed positions> ===== #

        # ===== <!!! Prompting !!!> =====#
        # concatenation: [p1, p2, p3, ..., pN] + [original tokens]
        # No positional embedding on prompts

        if self.prefix_prompt_length > 0:
            # prefix prompt
            prefix_task_embed_idx = torch.tensor(list(range(self.prefix_prompt_length))).cuda()
            prefix_task_embed = self.prefix_embed_prompts(prefix_task_embed_idx)
            prefix_task_embed = self.prompt_dropout(prefix_task_embed)

        if self.infix_prompt_length > 0:
            # infix prompt
            infix_task_embed_idx = torch.tensor(list(range(self.infix_prompt_length))).cuda()
            infix_task_embed = self.infix_embed_prompts(infix_task_embed_idx)
            infix_task_embed = self.prompt_dropout(infix_task_embed)

        # ===== <concat prompt> ===== #
        x_sep = None
        prompt_type = None
        # condition 1: prefix prompt and infix prompt at input
        if self.prefix_prompt_length > 0 and self.infix_prompt_length > 0:
            x_sep = (prev_output_tokens == 0).nonzero()
            x = torch.concat(
                [
                    torch.concat(
                        (prefix_task_embed, src[:sep_i], infix_task_embed, src[sep_i:]),
                        dim=0,
                    ).unsqueeze(0)
                    for src, (i, sep_i) in zip(x, x_sep)
                ]
            )
            prompt_type = "prefix_infix"

        # condition 2: prefix prompt at input
        elif self.prefix_prompt_length > 0 and self.infix_prompt_length == 0:
            x = torch.concat([torch.concat((prefix_task_embed, src), dim=0).unsqueeze(0) for src in x])
            prompt_type = "prefix"

        # condition 3: infix prompt at input
        elif self.prefix_prompt_length == 0 and self.infix_prompt_length > 0:
            x_sep = (prev_output_tokens == 0).nonzero()
            x = torch.concat(
                [
                    torch.concat(
                        (src[:sep_i], infix_task_embed, src[sep_i:]),
                        dim=0,
                    ).unsqueeze(0)
                    for src, (i, sep_i) in zip(x, x_sep)
                ]
            )
            prompt_type = "infix"

        # ===== </concat prompt> ===== #
        # dummy padding for attention padding mask
        pad = (
            torch.tensor(self.dictionary.bos_index)
            .unsqueeze(-1)
            .repeat(
                prev_output_tokens.size(0),
                self.prefix_prompt_length + self.infix_prompt_length,
            )
            .to(prev_output_tokens.device)
        )

        prev_output_tokens = torch.concat((pad, prev_output_tokens), dim=1)
        # ===== </!!! Prompting !!!> =====#

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        attn_layers = list()
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # save attn layers input
            attn_layers.append(x)

            # === <!!! deep prompt tuning!!!> === #
            if self.deep_prompt and prompt_type == "prefix":
                deep_key_prompt = self.deep_key_embed_prompts[idx](prefix_task_embed_idx)
                deep_value_prompt = self.deep_value_embed_prompts[idx](prefix_task_embed_idx)
                deep_key_prompt = self.prompt_dropout(deep_key_prompt)
                deep_value_prompt = self.prompt_dropout(deep_value_prompt)
            elif self.deep_prompt and prompt_type == "infix":
                deep_key_prompt = self.deep_key_embed_prompts[idx](infix_task_embed_idx)
                deep_value_prompt = self.deep_value_embed_prompts[idx](infix_task_embed_idx)
                deep_key_prompt = self.prompt_dropout(deep_key_prompt)
                deep_value_prompt = self.prompt_dropout(deep_value_prompt)
            else:
                deep_key_prompt = None
                deep_value_prompt = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                key_prompt=deep_key_prompt,
                value_prompt=deep_value_prompt,
                prompt_type=prompt_type,
                x_sep=x_sep,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            # === </!!! deep prompt tuning!!!> === #

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if src_lengths is not None:
            return x, {
                "embed": embed,
                "attn_layers": attn_layers,
                "attn": [attn],
                "inner_states": inner_states,
                "prompt_length": self.prefix_prompt_length + self.infix_prompt_length,
            }
        else:
            return x, {"embed": embed, "attn_layers": attn_layers, "attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to  vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            output_projection = self.output_projection(features)

            if self.linear_verbalizer and (self.prefix_prompt_length > 0 or self.infix_prompt_length > 0):
                # add prompt_linear_verbalizer with dropout and temperature
                out = self.prompt_linear_verbalizer(output_projection)
                out = self.prompt_verbalizer_dropout(out)
                # return out, output_projection[:, -1, :]
            else:
                out = output_projection

            return out
        else:
            return features

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = TransformerDecoderPromptLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class TransformerDecoderPrompt(TransformerDecoderPromptBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        prompts,
        embed_sep_token=None,
        verbalizer=None,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            prompts,
            embed_sep_token,
            verbalizer,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(TransformerConfig.from_namespace(args), dictionary, embed_tokens)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn)
