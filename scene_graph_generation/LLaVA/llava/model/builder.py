#    Copyright 2023 Haotian Liu
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


import os
import shutil
import warnings

import torch
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import *
from llava.model.language_model.llava_llama import LlavaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


def _drop_training_quantization(cfg):
    # QLoRA checkpoints save the bnb settings they were *trained* with in config.json.
    # transformers 4.31 ignored that when loading the base model, so the original
    # pipeline evaluated in bf16; 4.57 honors it and builds 4-bit layers that the
    # bf16 base checkpoint never fills (left on meta), i.e. it treats the bf16 base as an
    # already quantized checkpoint. Drop it always; load_8bit/load_4bit pass their own config.
    if getattr(cfg, "quantization_config", None) is not None:
        del cfg.quantization_config
    return cfg


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        # Unchanged from upstream and not verified on the modernized stack.
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # The QLoRA training setup (train.py): NF4 base with bf16 compute, projector and
        # pooler left unquantized. The vision tower is built separately during training,
        # so it is never quantized there either; here it is part of from_pretrained.
        # load_in_4bit lives only in quantization_config (newer transformers reject both).
        kwargs['torch_dtype'] = torch.bfloat16
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=["mm_projector", "image_pooler", "vision_tower"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            # LlavaConfig, not AutoConfig: released checkpoints say model_type "llava",
            # which transformers now resolves to its own native LlavaConfig.
            lora_cfg_pretrained = _drop_training_quantization(LlavaConfig.from_pretrained(model_path))
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

            # entities = ["head surgeon", "assistant surgeon", "circulator", "nurse", "anaesthetist", "patient", "instrument table", "operating table",
            #             "secondary table", "anesthesia equipment", "instrument"]
            # predicates = ["assisting", "cementing", "cleaning", "closeTo", "cutting", "drilling", "hammering", "holding", "lyingOn", "manipulating",
            #               "preparing", "sawing", "suturing", "touching"]
            # tokenizer.add_special_tokens({"additional_special_tokens": entities + predicates})

            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            # a 4-bit lm_head stores its weight packed, so its shape says nothing about the vocab
            if not load_4bit and model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu', weights_only=True)
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu', weights_only=True)

                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            if load_4bit:
                # Keep the adapters separate: the model then computes exactly what it was
                # trained as (LoRA on top of the NF4 base). Merging into a bf16 base gives a
                # slightly different model; see MODERNIZATION.md, "Evaluating in 4-bit".
                print('Keeping LoRA weights unmerged (4-bit base)...')
            else:
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = _drop_training_quantization(LlavaConfig.from_pretrained(model_path))
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu', weights_only=True)
            mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            non_lora_trainables = None
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.bfloat16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        # Only when tokens were actually added: resizing touches lm_head, which fails on
        # a 4-bit (non-floating-point) lm_head.
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor
        # if non_lora_trainables contains something about vision_tower, load it
        if non_lora_trainables is not None and any(k.startswith('model.vision_tower.') for k in non_lora_trainables):
            new_vision_tower_state_dict = {}
            for k, v in non_lora_trainables.items():  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
                if 'model.vision_tower.vision_tower.' in k:
                    new_k = k.replace('model.vision_tower.', '')
                    new_vision_tower_state_dict[new_k] = v
            print('Loaded additional vision tower weights...')
            vision_tower.load_state_dict(new_vision_tower_state_dict, strict=False)
            # weight difference sum([torch.norm(value-vision_tower.state_dict()[key].cpu())  for key,value in new_vision_tower_state_dict.items()])
        image_pooler = model.get_image_pooler()
        image_pooler.to(device=device, dtype=torch.bfloat16)
        if non_lora_trainables is not None and any(k.startswith('model.image_pooler.') for k in non_lora_trainables):
            new_image_pooler_state_dict = {}
            for k, v in non_lora_trainables.items():  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
                if 'model.image_pooler.' in k:
                    new_k = k.replace('model.image_pooler.', '')
                    new_image_pooler_state_dict[new_k] = v
            print('Loading additional image pooler weights...')
            # solve some compatibility issues in transformers
            new_image_pooler_state_dict.pop('bert.embeddings.position_ids', None)
            new_image_pooler_state_dict.pop('bert.embeddings.token_type_ids', None)
            image_pooler.load_state_dict(new_image_pooler_state_dict, strict=True)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
