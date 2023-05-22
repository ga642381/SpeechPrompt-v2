import json
import math
from ast import arg
from dataclasses import dataclass, field
from lib2to3.pgen2 import token
from typing import Any, Dict, List, Optional

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerConfig, TransformerDecoderBase
from fairseq.models.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig, transformer_lm_big
from fairseq.modules import SinusoidalPositionalEmbedding
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.utils import safe_getattr, safe_hasattr

from .transformer_prompt_decoder import TransformerDecoderPrompt

# from fairseq_usr.transformer import TransformerDecoderPrompt
from torch import Tensor
from torch.nn import Dropout, ModuleList

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class Prompts:
    deep_prompt: bool
    prompt_length: int
    dec_input_prompt: torch.nn.Embedding
    dec_key_prompt: torch.nn.ModuleList
    dec_value_prompt: torch.nn.ModuleList


@dataclass
class Verbalizer:
    num_classes: int
    linear_verbalizer: torch.nn.Linear


@dataclass
class GSLMPromptConfig(TransformerLanguageModelConfig):
    prompt_length: int = field(default=10, metadata={"help": "prefix prompt length"})
    deep_prompt: bool = field(default=True, metadata={"help": "if using deep prompt"})
    linear_verbalizer: bool = field(default=True, metadata={"help": "if using linear verbalizer"})
    num_classes: int = field(default=2, metadata={"help": "number of classes in classification task"})
    use_sep_token: bool = field(default=True, metadata={"help": "if using sep token <s> "})
    fine_tune: bool = field(default=False, metadata={"help": "if finetuning the whole model"})
    verbalizer: Optional[Verbalizer] = None
    prompts: Optional[Prompts] = None


@register_model("GSLM_prompt", dataclass=GSLMPromptConfig)
class GSLMPromptModel(TransformerLanguageModel):
    def __init__(self, cfg: GSLMPromptConfig, decoder):
        super().__init__(decoder)
        self.cfg = cfg

    @classmethod
    def build_model(cls, args, task):
        ######################
        #   Token Embedding  #
        ######################
        embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)

        ##############
        #   Prompts  #
        ##############
        if args.prompt_length > 0:
            prompts = cls.build_prompt_embedding(
                args,
                args.decoder_embed_dim,
                args.decoder_layers,
                prompt_length=args.prompt_length,
                deep_prompt=args.deep_prompt,
            )
        else:
            # prompts = Prompts(None, None, None, None, None)
            prompts = None

        ######################
        #    Sep Embedding   #
        ######################
        if args.use_sep_token:
            sep_embed_token = cls.build_sep_embedding(args, args.decoder_embed_dim)
        else:
            sep_embed_token = None

        ##############
        # Verbalzier #
        ##############
        if args.linear_verbalizer and not args.fine_tune:
            verbalizer = cls.build_linear_verbalizer(args, args.num_classes, task.source_dictionary)
        else:
            # verbalizer = Verbalizer(None, None)
            verbalizer = None

        ###############
        # Build Model #
        ###############
        decoder = TransformerDecoderPrompt(
            args, task.target_dictionary, embed_tokens, prompts, sep_embed_token, verbalizer, no_encoder_attn=True
        )

        if not args.fine_tune:
            # ===== <!!! Prompting !!!> =====#
            # fix the pretrained model parameters
            # make prompt parameters trainable
            # make sure these trainable parameters will (only) be saved when saving prompt checkpoint
            for name, p in decoder.named_parameters():
                p.requires_grad = False
                if "prompt" in name:
                    p.requires_grad = True
                if "sep" in name:
                    p.requires_grad = True
                if "verbalizer" in name:
                    p.requires_grad = True

        return cls(args, decoder)

    @classmethod
    def build_sep_embedding(cls, args, embed_dim, path=None):
        """
        Build a new seperation token embedding.
        We make the seperation token embedding trainable, serving as a infix-prompt token.
        """
        sep_embed = torch.nn.Embedding(1, embed_dim, padding_idx=None)
        return sep_embed

    @classmethod
    def build_linear_verbalizer(cls, args, num_classes, dictionary):
        """
        Build a new verbalizer
        The input dimension is the GSLM's vocabulary size, and the output dimension is the number of classes + 4.
        4 means the number of special tokens: <s>, </s>, <pad>, <unk>
        """
        linear_verbalizer = torch.nn.Linear(len(dictionary), num_classes + 4, bias=False)
        return Verbalizer(num_classes, linear_verbalizer)

    @classmethod
    def build_prompt_embedding(cls, args, dec_embed_dim, decoder_layers, prompt_length, deep_prompt=False):
        """
        Build prompt embeddings for GSLM:
        1. decoder input prompt: [prompt_length]
        2. decoder key prompt: [decoder_layers x prompt_length]
        3. decoder value prompt: [decoder_layers x prompt_length]
        """
        #############
        #  Dec Only #
        #############
        # === input prompt === #
        dec_input_prompt = torch.nn.Embedding(prompt_length, dec_embed_dim, padding_idx=None)

        # === deep prompt === #
        if deep_prompt:
            dec_key_prompt = ModuleList(
                [torch.nn.Embedding(prompt_length, dec_embed_dim, padding_idx=None) for i in range(decoder_layers)]
            )
            dec_value_prompt = ModuleList(
                [torch.nn.Embedding(prompt_length, dec_embed_dim, padding_idx=None) for i in range(decoder_layers)]
            )
        else:
            dec_key_prompt = None
            dec_value_prompt = None

        return Prompts(deep_prompt, prompt_length, dec_input_prompt, dec_key_prompt, dec_value_prompt)


@register_model_architecture("GSLM_prompt", "GSLM_SpeechPrompt_v1")
def GSLM_SpeechPrompt_v1(args):
    """
    SpeechPrompt v1
    1. Prefix-prompt length can be adjusted.
    2. Both deep prompt tuning and input prompt tuning are studied.
    2. There's no linear verbalizer.
    """
    args.prompt_length = safe_getattr(args, "prompt_length", 10)
    args.deep_prompt = safe_getattr(args, "deep_prompt", True)
    args.linear_verbalizer = safe_getattr(args, "linear_verbalizer", False)
    args.num_classes = safe_getattr(args, "num_classes", 0)
    args.use_sep_token = safe_getattr(args, "use_sep_token", True)
    args.fine_tune = safe_getattr(args, "fine_tune", False)

    transformer_lm_big(args)


@register_model_architecture("GSLM_prompt", "GSLM_prompt_v2")
def GSLM_SpeechPrompt_v2(args):
    """
    SpeechPrompt v2
    1. Preifx-prompt length can be adjusted. We used prompt length 5 in the paper.
    2. There's is a trainable linear verbalizer.
    """
    args.prompt_length = safe_getattr(args, "prompt_length", 5)
    args.deep_prompt = safe_getattr(args, "deep_prompt", True)
    args.linear_verbalizer = safe_getattr(args, "linear_verbalizer", True)
    args.num_classes = safe_getattr(args, "num_classes", 2)
    args.use_sep_token = safe_getattr(args, "use_sep_token", True)
    args.fine_tune = safe_getattr(args, "fine_tune", False)

    transformer_lm_big(args)


@register_model_architecture("GSLM_prompt", "GSLM_prompt_finetune")
def GSLM_SpeechPrompt_finetune(args):
    """
    SpeechPrompt Fine-tune
    1. Preifx-prompt length can be adjusted.
    2. The whole GSLM is fine-tuned.
    """
    args.prompt_length = safe_getattr(args, "prompt_length", 0)
    args.deep_prompt = safe_getattr(args, "deep_prompt", True)
    args.linear_verbalizer = safe_getattr(args, "linear_verbalizer", False)
    args.num_classes = safe_getattr(args, "num_classes", 0)
    args.use_sep_token = safe_getattr(args, "use_sep_token", True)
    args.fine_tune = safe_getattr(args, "fine_tune", True)
    transformer_lm_big(args)
