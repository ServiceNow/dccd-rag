import os
import sys

import pytest

# Make the repository root importable without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import build_tiny_checkpoint  # noqa: E402


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory) -> str:
    """A tiny random-init causal LM + word-level tokenizer saved to disk."""
    return build_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_ckpt")))


@pytest.fixture(scope="session")
def tiny_model(tiny_ckpt):
    """``(model, tokenizer)`` for :func:`tiny_ckpt`, loaded once per session."""
    from dccd.models import load_model_and_tokenizer
    return load_model_and_tokenizer(tiny_ckpt, dtype="float32", device_map="cpu")
