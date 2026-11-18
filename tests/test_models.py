"""Model loading: attention-kernel selection and the no-flash-attn fallback."""
import importlib.util

import pytest
import torch

from dccd import models
from helpers import build_tiny_checkpoint

_real_find_spec = importlib.util.find_spec


def _fake_find_spec(flash_attn_present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "flash_attn":
            return object() if flash_attn_present else None
        return _real_find_spec(name, *args, **kwargs)
    return find_spec


@pytest.fixture(scope="module")
def small_head_ckpt(tmp_path_factory):
    return build_tiny_checkpoint(str(tmp_path_factory.mktemp("hd64")), head_dim=64)


def test_select_eager_for_large_head_dim(tiny_ckpt):
    assert models.select_attn_implementation(tiny_ckpt) == "eager"


def test_select_sdpa_without_flash_attn(small_head_ckpt, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(False))
    assert models.select_attn_implementation(small_head_ckpt) == "sdpa"


def test_select_fa2_when_available(small_head_ckpt, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert models.select_attn_implementation(small_head_ckpt) == "flash_attention_2"


def test_loads_on_cpu_without_flash_attn(small_head_ckpt):
    model, tokenizer = models.load_model_and_tokenizer(
        small_head_ckpt, dtype="float32", device_map="cpu")
    assert model.config._attn_implementation == "sdpa"
    assert tokenizer.padding_side == "left"


def test_explicit_attn_implementation_respected(small_head_ckpt):
    model, _ = models.load_model_and_tokenizer(
        small_head_ckpt, dtype="float32", device_map="cpu", attn_implementation="eager")
    assert model.config._attn_implementation == "eager"


def test_loads_hub_style_longrope_config(tmp_path):
    """Phi-3-medium-128k's Hub config fails on transformers>=5 unless
    original_max_position_embeddings is moved into the RoPE parameters."""
    from helpers import build_tiny_longrope_checkpoint
    path = build_tiny_longrope_checkpoint(str(tmp_path / "phi3"))
    model, _ = models.load_model_and_tokenizer(path, dtype="float32", device_map="cpu")
    assert type(model).__name__ == "Phi3ForCausalLM"
    assert model.config.original_max_position_embeddings == 4096


def _record_trust(monkeypatch):
    """Patch every from_pretrained the loader uses to record trust_remote_code."""
    import transformers
    seen = []
    for cls in (transformers.AutoTokenizer, transformers.AutoModelForCausalLM,
                transformers.AutoConfig):
        real = cls.from_pretrained

        def wrapped(*args, _real=real, **kwargs):
            seen.append(kwargs.get("trust_remote_code"))
            return _real(*args, **kwargs)
        monkeypatch.setattr(cls, "from_pretrained", wrapped)
    return seen


def test_remote_code_off_by_default(small_head_ckpt, monkeypatch):
    seen = _record_trust(monkeypatch)
    models.load_model_and_tokenizer(small_head_ckpt, dtype="float32", device_map="cpu")
    assert seen and not any(seen)


def test_remote_code_opt_in(small_head_ckpt, monkeypatch):
    seen = _record_trust(monkeypatch)
    models.load_model_and_tokenizer(small_head_ckpt, dtype="float32", device_map="cpu",
                                    trust_remote_code=True)
    assert seen and all(seen)
