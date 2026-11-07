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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# MM2SG's model type. Renamed from "llava" because transformers (>=4.36) ships its own
# native LLaVA registered under exactly that name, so `AutoConfig.register("llava", ...)`
# now raises "'llava' is already used by a Transformers config".
#
# The authors' released checkpoints carry "model_type": "llava" in config.json. They still
# load, because every load site constructs LlavaConfig explicitly instead of going through
# AutoConfig -- see llava/model/builder.py. PretrainedConfig.from_pretrained ignores a
# mismatched model_type when the class is named directly.
LLAVA_MM2SG_MODEL_TYPE = "llava_mm2sg"


class LlavaConfig(LlamaConfig):
    model_type = LLAVA_MM2SG_MODEL_TYPE


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        vis_descriptor_embs=None,
        pc=None,
        audio=None,
        segmasks=None,
        # transformers >=4.38 generate() passes cache_position (and later versions pass
        # further kwargs such as logits_to_keep). The signature must accept them or
        # generate() dies with "unexpected keyword argument". **kwargs keeps this
        # forward-compatible instead of chasing each new argument.
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        spliced_multimodal = False
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                vis_descriptor_embs,
                pc,
                audio,
                segmasks
            )
            # prepare_inputs_labels_for_multimodal splices modality embeddings into the
            # sequence, so the real sequence -- and hence the KV cache -- is longer than the
            # text-only input_ids generate() derives cache_position from. That holds for the
            # prefill AND for every later decode step (the cache keeps the spliced length).
            # A stale cache_position builds a causal mask that hides most of the context
            # from each new token; it only bites when an explicit mask is materialized,
            # i.e. when the batch contains padding (then every row in the batch is garbage).
            # Drop it and let the base model recompute it from past_key_values.
            spliced_multimodal = images is not None

        extra = dict(kwargs)
        if cache_position is not None and not spliced_multimodal:
            extra["cache_position"] = cache_position
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **extra
        )
        output['modified_labels'] = labels
        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        pc = kwargs.pop("pc", None)
        audio = kwargs.pop("audio", None)
        segmasks = kwargs.pop("segmasks", None)
        vis_descriptor_embs = kwargs.pop("vis_descriptor_embs", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if vis_descriptor_embs is not None:
            _inputs['vis_descriptor_embs'] = vis_descriptor_embs
        if pc is not None:
            _inputs['pc'] = pc
        if audio is not None:
            _inputs['audio'] = audio
        if segmasks is not None:
            _inputs['segmasks'] = segmasks
        return _inputs


AutoConfig.register(LLAVA_MM2SG_MODEL_TYPE, LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
