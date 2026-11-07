"""
Dtype fixups for k-bit (QLoRA) training.

Extracted from the deleted llava/train/llama_patch.py. That file existed to monkeypatch
LlamaAttention for FlashAttention -- obsolete now that transformers selects attention
natively via `attn_implementation` -- but this helper is unrelated to attention and is
still required by llava/train/train.py after prepare_model_for_kbit_training().
"""

from peft.tuners.lora import LoraLayer


def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model
