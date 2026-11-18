"""Model and tokenizer loading.

Reproduces the paper runner's setup (``contrastive_decoding_baselines.py``):
bfloat16, left padding, and the same attention-kernel choice on the paper's
hardware -- ``flash_attention_2`` for head_dim <= 256, else ``eager``. The
attention kernel affects bf16 numerics, so matching it matters for reproducing
the published results bit-for-bit.

On machines without ``flash-attn`` or without a CUDA GPU (laptops, Colab CPU,
Apple silicon), ``sdpa`` is used instead, so the code runs everywhere; outputs
then match the paper's metrics up to small bf16 differences.
"""
from __future__ import annotations

import importlib.util
from typing import Optional

import torch
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoTokenizer,
                          PretrainedConfig)


def _load_config(model_path: str, trust_remote_code: bool):
    """Return ``(config, patched)``.

    Phi-3 128k checkpoints on the Hub (e.g. Phi-3-medium-128k-instruct) keep
    ``original_max_position_embeddings`` at the top level of ``config.json``;
    transformers>=5 requires it inside the long-RoPE parameters and refuses to
    load the config. We copy it there in memory -- the same fix the paper's
    zero-shot runner applied to its cached config file.
    """
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code), False
    except KeyError as e:
        if "original_max_position_embeddings" not in str(e):
            raise
    config_dict, _ = PretrainedConfig.get_config_dict(model_path)
    rope = config_dict.get("rope_scaling")
    if isinstance(rope, dict):
        rope.setdefault("original_max_position_embeddings",
                        config_dict.get("original_max_position_embeddings", 4096))
    return AutoConfig.for_model(config_dict.pop("model_type"), **config_dict), True


def _max_head_dim(model_path: str, trust_remote_code: bool = False) -> Optional[int]:
    try:
        cfg, _ = _load_config(model_path, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    head_dims = []
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is None:
            continue
        for attr in ("head_dim", "global_head_dim", "qk_head_dim", "v_head_dim"):
            v = getattr(c, attr, None)
            if v is not None:
                head_dims.append(int(v))
    return max(head_dims) if head_dims else None


def select_attn_implementation(model_path: str, trust_remote_code: bool = False) -> str:
    """``eager`` for head_dim > 256 (as the paper runner); otherwise
    ``flash_attention_2`` when flash-attn and a CUDA GPU are available, else
    ``sdpa``."""
    max_head_dim = _max_head_dim(model_path, trust_remote_code)
    if max_head_dim and max_head_dim > 256:
        return "eager"
    if importlib.util.find_spec("flash_attn") is not None and torch.cuda.is_available():
        return "flash_attention_2"
    return "sdpa"


def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16",
                             device_map: str = "auto",
                             attn_implementation: Optional[str] = None,
                             trust_remote_code: bool = False):
    """Load ``(model, tokenizer)`` matching the paper runner's configuration.

    ``attn_implementation`` overrides the automatic choice
    (:func:`select_attn_implementation`). ``trust_remote_code`` lets a Hub
    checkpoint run its own modelling code; it is off by default because none of
    the paper's models need it (Qwen3.5 has no remote code, and Phi-3 must use
    the upstream implementation).
    """
    torch_dtype = getattr(torch, dtype)

    # Phi-3 ships bundled remote code incompatible with the current transformers
    # cache API; always use the upstream implementation, as the paper runner did.
    trust = bool(trust_remote_code) and "Phi-3" not in model_path
    config, patched = _load_config(model_path, trust_remote_code=trust)
    extra = {"config": config} if patched else {}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust, **extra)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = attn_implementation or select_attn_implementation(model_path, trust)
    load_errors = (ImportError, ValueError, TypeError)

    def _load(cls, impl):
        return cls.from_pretrained(
            model_path, dtype=torch_dtype, device_map=device_map,
            trust_remote_code=trust, attn_implementation=impl, **extra)

    try:
        model = _load(AutoModelForCausalLM, attn_impl)
    except load_errors as e:
        if "flash" in str(e).lower():
            # Requested kernel unavailable: fall back to sdpa, then eager.
            try:
                model = _load(AutoModelForCausalLM, "sdpa")
            except load_errors:
                try:
                    model = _load(AutoModelForCausalLM, "eager")
                except load_errors:
                    model = _load(AutoModelForImageTextToText, "eager")
        else:
            # Multimodal checkpoints (e.g. Qwen3.5) load through the
            # image-text-to-text auto class, as in the paper runner.
            model = _load(AutoModelForImageTextToText, attn_impl)

    model.eval()
    return model, tokenizer


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool = False,
                        no_chat_template: bool = False) -> str:
    """Wrap a raw user prompt with the tokenizer's chat template (thinking off
    by default), matching the paper runner."""
    if no_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
