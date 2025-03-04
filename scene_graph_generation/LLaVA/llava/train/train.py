# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import json
import logging
import math
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np
import open3d as o3d
import torch
import transformers
from PIL import Image
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer
from torch import Tensor
from torch.utils.data import Dataset
from torchinfo import summary
from torchvision.transforms import functional as F, InterpolationMode

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mv_type: Optional[str] = field(default='concat')
    use_vis_descriptors: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    token_weight_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    do_augment: bool = field(default=False)
    do_img_order_augment: bool = field(default=False)
    do_multimodal_augment: bool = field(default=False)
    multimodal_drop_prop: float = field(default=0.5)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_n_vision_tower_layers: Optional[int] = field(default=None)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    curriculum_learning_weights: Optional[str] = field(default=None)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    raise Exception('The extended version should be used instead.')
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3_extended(model, require_grad_only=True):
    '''
    Supports both params but also buffers
    '''
    named_entities = list(model.named_parameters()) + list(model.named_buffers())
    to_return = {k: v for k, v in named_entities if "lora_" not in k}
    if require_grad_only:
        # For buffers, requires_grad attribute does not apply, so they should be included regardless
        to_return = {k: v for k, v in to_return.items() if type(v) == torch.Tensor or v.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'image_pooler']  # might need expanding
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            value = sentence['value']
            conv.append_message(role, value)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None, strength: float = 1.0) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.strength = max(0.0, min(strength, 1.0))  # Ensuring strength is within [0, 1]

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        scale_factor = self.strength
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0 * scale_factor, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0 * scale_factor, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0 * scale_factor, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99 * scale_factor, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.do_img_order_augment = self.data_args.do_img_order_augment
        self.do_multimodal_augment = self.data_args.do_multimodal_augment
        self.multimodal_drop_prop = self.data_args.multimodal_drop_prop
        if self.data_args.do_augment:
            self.augment = TrivialAugmentWide(strength=0.5)  # less augment
        else:
            self.augment = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        potentially_included_modalities = set()
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']

            if self.do_img_order_augment:
                synthetic_supporting_images = []
                contains_synthetic_example = 'synthetic_mv_hybridor' in str(image_file)
                if contains_synthetic_example:
                    dataset_type = '4D-OR' if '4DOR' in sources[0]['id'] else 'MM-OR'
                    synthetic_supporting_views = (2, 1, 3, 5) if dataset_type == '4D-OR' else (1, 4, 5)
                    for v_idx in synthetic_supporting_views:
                        for img_file in image_file:
                            if f'_cidx{v_idx}' in img_file and 'synthetic' in img_file:
                                synthetic_supporting_images.append(img_file)

                random.shuffle(image_file)
                # n_images = random.randint(1, len(image_file)) # we can not support up to 10 images with batchsize 4. We need to go to batchsize 2 in that case. Instead we can simply use 7 images
                n_images = random.randint(1, min(7, len(image_file)))
                image_file = image_file[:n_images]
                # if synthetic sample we have to include some images from azure we can't fully randomly sample. Also important which azure images we include
                if contains_synthetic_example:
                    # check that we included at least one synthetic supporting image file. If not, add one randomly. check if anything in the list synthetic_supporting_images is in image_file
                    if not any([img_file in image_file for img_file in synthetic_supporting_images]) and len(synthetic_supporting_images) > 0:
                        # replace instead of adding, because max is 7.
                        image_file[-1] = random.choice(synthetic_supporting_images)

                image_file_str = ''.join(image_file)
                if 'colorimage' in image_file_str:
                    potentially_included_modalities.add('azure')
                if 'simstation' in image_file_str:
                    potentially_included_modalities.add('simstation')
                if 'trackercam' in image_file_str:
                    potentially_included_modalities.add('trackercam')

            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if type(image_file) == list:
                image = [Image.open(os.path.join(image_folder, image_file)).convert('RGB') for image_file in image_file]
                # augment images
                if self.augment is not None:
                    image = [self.augment(image) for image in image]
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [expand2square(image, tuple(int(x * 255) for x in processor.image_mean)) for image in image]
                    image = [processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
                    # stack images
                    image = torch.stack(image, dim=0)
                else:
                    image = [processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.augment is not None:
                    image = self.augment(image)
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        # decide if we keep robot and tracker metadata
        if self.do_multimodal_augment and random.random() < self.multimodal_drop_prop:  # we are deleting robot_metadata
            # robot_metadata starts with <robot_metadata_start>, ends with <robot_metadata_end>. Find those and delete between them
            inp_str = sources[0][0]['value']
            robot_metadata_start = '<robot_metadata_start>'
            robot_metadata_end = '<robot_metadata_end>. '
            robot_metadata_start_idx = inp_str.find(robot_metadata_start)
            robot_metadata_end_idx = inp_str.find(robot_metadata_end)
            if robot_metadata_start_idx != -1 and robot_metadata_end_idx != -1:
                inp_str = inp_str[:robot_metadata_start_idx] + inp_str[robot_metadata_end_idx + len(robot_metadata_end):]
                sources[0][0]['value'] = inp_str
        else:
            potentially_included_modalities.add('robot_metadata')
        # same with tracker data
        if self.do_multimodal_augment and random.random() < self.multimodal_drop_prop:  # we are deleting tracker_metadata
            # tracker_metadata starts with <tracker_metadata_start>, ends with <tracker_metadata_end>. Find those and delete between them
            inp_str = sources[0][0]['value']
            tracker_metadata_start = '<tracker_metadata_start>'
            tracker_metadata_end = '<tracker_metadata_end>. '
            tracker_metadata_start_idx = inp_str.find(tracker_metadata_start)
            tracker_metadata_end_idx = inp_str.find(tracker_metadata_end)
            if tracker_metadata_start_idx != -1 and tracker_metadata_end_idx != -1:
                inp_str = inp_str[:tracker_metadata_start_idx] + inp_str[tracker_metadata_end_idx + len(tracker_metadata_end):]
                sources[0][0]['value'] = inp_str
        else:
            potentially_included_modalities.add('tracker_metadata')
        # same with speech transcript
        if self.do_multimodal_augment and random.random() < self.multimodal_drop_prop:  # we are deleting speech_transcript
            # speech_transcript starts with <speech_transcript_start>, ends with <speech_transcript_end>. Find those and delete between them
            inp_str = sources[0][0]['value']
            speech_transcript_start = '<speech_transcript_start>'
            speech_transcript_end = '<speech_transcript_end>. '
            speech_transcript_start_idx = inp_str.find(speech_transcript_start)
            speech_transcript_end_idx = inp_str.find(speech_transcript_end)
            if speech_transcript_start_idx != -1 and speech_transcript_end_idx != -1:
                inp_str = inp_str[:speech_transcript_start_idx] + inp_str[speech_transcript_end_idx + len(speech_transcript_end):]
                sources[0][0]['value'] = inp_str
        else:
            potentially_included_modalities.add('speech_transcript')

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
        if 'vis_knowledge_paths' in self.list_data_dict[i] and self.list_data_dict[i]['vis_knowledge_paths'] is not None:
            vis_descriptor_embs = []
            for vis_knowledge_path in self.list_data_dict[i]['vis_knowledge_paths']:
                vis_knowledge_path = os.path.join(os.path.dirname(os.getcwd()), vis_knowledge_path)
                emb = torch.load(vis_knowledge_path, map_location='cpu')
                vis_descriptor_embs.append(emb)
            data_dict['vis_descriptor_embs'] = vis_descriptor_embs
        if 'pc' in self.list_data_dict[i]:
            if not self.do_multimodal_augment or random.random() > self.multimodal_drop_prop:  # maybe include
                pc_path = self.list_data_dict[i]['pc']
                pc = o3d.io.read_point_cloud(pc_path)
                pc = torch.from_numpy(np.concatenate([np.asarray(pc.points) / 1000, np.asarray(pc.colors)], axis=-1)).float()
                data_dict['pc'] = pc
                potentially_included_modalities.add('pc')
        if 'audio' in self.list_data_dict[i]:
            if not self.do_multimodal_augment or random.random() > self.multimodal_drop_prop:  # maybe include
                audio_embeeding_path = self.list_data_dict[i]['audio']
                data_dict['audio'] = torch.load(audio_embeeding_path, map_location='cpu')
                potentially_included_modalities.add('audio')
        if 'segmasks' in self.list_data_dict[i]:
            # they are one channel images, read them accordingly
            if not self.do_multimodal_augment or random.random() > self.multimodal_drop_prop:  # maybe include
                segmasks = [torch.from_numpy(np.array(Image.open(segmask_path).convert('L'))) for segmask_path in self.list_data_dict[i]['segmasks']]
                data_dict['segmasks'] = segmasks
                potentially_included_modalities.add('segmasks')
        # if 'MM-OR' in str(image_file):
        #     print(potentially_included_modalities)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'vis_descriptor_embs' in instances[0]:
            vis_descriptor_embs = [instance['vis_descriptor_embs'] for instance in instances]
            batch['vis_descriptor_embs'] = vis_descriptor_embs

        # check if any instance contains 'pc'
        if any('pc' in instance for instance in instances):
            # still it is possible that some instances do not have 'pc', those shall be None instead
            pc = [instance['pc'] if 'pc' in instance else None for instance in instances]
            batch['pc'] = pc
        # check if any instance contains 'audio'
        if any('audio' in instance for instance in instances):
            # still it is possible that some instances do not have 'audio', those shall be None instead
            audio = [instance['audio'] if 'audio' in instance else None for instance in instances]
            batch['audio'] = audio

        # check if any instance contains 'segmasks'
        if any('segmasks' in instance for instance in instances):
            # still it is possible that some instances do not have 'segmasks', those shall be None instead
            segmasks = [instance['segmasks'] if 'segmasks' in instance else None for instance in instances]
            batch['segmasks'] = segmasks

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_model_weights(ref_model_path, peft_model, device):
    # Store the current requires_grad status
    original_requires_grad = {name: param.requires_grad for name, param in peft_model.named_parameters()}

    non_lora_trainables = torch.load(os.path.join(ref_model_path, 'non_lora_trainables.bin'), map_location='cpu')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

    peft_model.base_model.model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    ref_model = PeftModel.from_pretrained(copy.deepcopy(peft_model.base_model.model), ref_model_path)
    peft_model.load_state_dict(ref_model.state_dict(), strict=False)
    del ref_model

    vision_tower = peft_model.base_model.model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    # if non_lora_trainables contains something about vision_tower, load it
    if non_lora_trainables is not None and any(k.startswith('model.vision_tower.') for k in non_lora_trainables):
        new_vision_tower_state_dict = {}
        for k, v in non_lora_trainables.items():  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
            if 'model.vision_tower.vision_tower.' in k:
                new_k = k.replace('model.vision_tower.', '')
                new_vision_tower_state_dict[new_k] = v
        print('Loaded additional vision tower weights...')
        vision_tower.load_state_dict(new_vision_tower_state_dict, strict=False)

    for name, param in peft_model.named_parameters():
        param.requires_grad = original_requires_grad[name]


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "image_pooler"],  # might need to add more
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )

    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.config.mv_type = model_args.mv_type

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # entities =  ["head surgeon", "assistant surgeon", "circulator", "nurse", "anaesthetist", "patient", "instrument table", "operating table", "secondary table", "anesthesia equipment", "instrument"]
        # predicates = ["assisting", "cementing", "cleaning", "closeTo", "cutting", "drilling", "hammering", "holding", "lyingOn", "manipulating", "preparing", "sawing", "suturing", "touching"]
        #
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict={"additional_special_tokens": entities + predicates},
        #     tokenizer=tokenizer,
        #     model=model,
        # )
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # reinitialize image_pooler # maybe more models like this
        model.get_model().image_pooler.bert = model.get_model().image_pooler.bert.apply(model.get_model().image_pooler.bert._init_weights)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            model.get_model().image_pooler.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        if training_args.unfreeze_n_vision_tower_layers is not None:
            print(f'Unfreezing last {training_args.unfreeze_n_vision_tower_layers} layers of vision tower')
            for layer in model.get_vision_tower().vision_tower.vision_model.encoder.layers[-training_args.unfreeze_n_vision_tower_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    if training_args.curriculum_learning_weights is not None:
        print(f'Initializing curriculum learning from {training_args.curriculum_learning_weights}')
        load_model_weights(training_args.curriculum_learning_weights, model, training_args.device)

    # save callback
    from transformers import TrainerCallback
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            if args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_extended(
                    model
                )
                if args.local_rank in [-1, 0]:
                    model.config.save_pretrained(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))

    summary(model, depth=5, col_names=['num_params', 'trainable'])

    from llava.train.llama_patch import upcast_layer_for_flash_attention
    model = upcast_layer_for_flash_attention(model, torch.bfloat16)
    if model_args.vision_tower is not None:
        print('Reinit and upcast vision towers to bfloat16')
        model.get_model().image_pooler.point_transformer._init_weights(dtype=torch.float32, device=training_args.device)
        model.get_model().image_pooler.segmasks_encoder._init_weights(dtype=torch.bfloat16, device=training_args.device)
    else:
        print('Text Only mode, no vision tower to reinit')

    if data_args.token_weight_path is not None:
        with open(data_args.token_weight_path, 'r') as f:
            token_frequencies = json.load(f)
    else:
        token_frequencies = None

    if token_frequencies is not None:
        # Linear weighting
        # token_weights = {k: 1 / v for k, v in token_frequencies.items()}
        # Log weighting (optional, uncomment if preferred)
        token_weights = {k: 1 / (np.log(v) + 1) for k, v in token_frequencies.items()}  # With this much unbalancing, log weighting is better
        min_weight = min(token_weights.values())
        extra_token_weight = min_weight / 100  # 100 times smaller than the smallest weight
    else:
        # If token_frequencies is None, skip these calculations
        token_weights = None
        min_weight = None
        extra_token_weight = None

    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           callbacks=[SaveCallback()],  # Makes sure to save additional stuff.
                           **data_module)
    trainer.extra_token_weight = extra_token_weight
    trainer.token_weights = token_weights
    trainer.min_weight = min_weight

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_extended(
            model
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
