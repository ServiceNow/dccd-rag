"""Test helpers: an offline tiny causal LM + word-level tokenizer, and examples.

``build_tiny_checkpoint`` writes a randomly initialised ``LlamaForCausalLM``
and a ``PreTrainedTokenizerFast`` to disk so tests (and CI) exercise the real
decoding loop without downloading anything. ``head_dim`` defaults to 320
(> 256), which makes every model loader in this repo pick ``eager`` attention,
so no flash-attn is needed.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from dccd.data import Document, Example
from dccd.prompts import (CONTEXTUAL_PROMPT, NO_VERBALIZERS, SUPPORT_PROBE_PROMPT,
                          YES_VERBALIZERS, ZEROSHOT_PROMPT)

SPECIAL_TOKENS = ["<unk>", "<|eos|>", "<|user|>", "<|assistant|>"]
THINK_TOKENS = ["<think>", "</think>"]
CHAT_TEMPLATE = ("{% for m in messages %}<|user|> {{ m['content'] }} {% endfor %}"
                 "<|assistant|>")

# Words used by the example factories below (plus filler for generation).
BASE_WORDS = """
the a an of in on at to and or is was are be by for with from that this it as
company maker makes made car cars model Civic Honda Toyota Ford answer question
document documents report email chat date revenue growth percent percentage
quarter year Q1 Q2 Q3 Q4 2023 2024 2025 2026 million rate customers online
recommendations increase decrease average order value reduced waste saving
Lee Market store stores team said says reported according current previous
not unknown maybe likely approximately about around which what who when where
Jason Wong IT helpdesk phishing incident March response time minutes contained
simulated coordinated quickly threat internal reducing under
""".split()


def _pieces(texts: Iterable[str]) -> List[str]:
    pre = pre_tokenizers.Whitespace()
    out = []
    for text in texts:
        out.extend(piece for piece, _ in pre.pre_tokenize_str(text))
    return out


def build_tiny_checkpoint(out_dir: str, head_dim: int = 320,
                          vocab_words: Optional[List[str]] = None) -> str:
    """Write a tiny random-init causal LM + word-level tokenizer to ``out_dir``."""
    texts = [CONTEXTUAL_PROMPT, ZEROSHOT_PROMPT, SUPPORT_PROBE_PROMPT,
             "Document [10](Title: Doc 10)", " ".join(BASE_WORDS),
             " ".join(str(i) for i in range(100)), ". , : ; ? ! ( ) [ ] % $ -"]
    texts += list(YES_VERBALIZERS) + list(NO_VERBALIZERS) + list(vocab_words or [])
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for piece in _pieces(texts):
        vocab.setdefault(piece, len(vocab))

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<|eos|>",
        pad_token="<|eos|>", additional_special_tokens=["<|user|>", "<|assistant|>"])
    tokenizer.add_tokens(THINK_TOKENS)
    tokenizer.chat_template = CHAT_TEMPLATE

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=head_dim, max_position_embeddings=32768, initializer_range=0.3,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None, tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def make_qa_example(n_docs: int = 5) -> Example:
    """A short multi-document QA example (NQ-style titles)."""
    texts = [
        "Honda is the company that makes the Civic car model .",
        "Toyota makes cars , according to the report .",
        "The Ford team said the answer is not current .",
        "Lee Market reported revenue growth of 8 percent in Q2 2024 .",
        "Which company makes the Civic ? The answer is Honda .",
        "The report said the rate was about 12 percent in 2023 .",
        "Online customers made recommendations in March 2025 .",
        "The previous model was made in 2023 by Toyota .",
        "An email said the answer is unknown .",
        "The chat said Honda makes the Civic .",
    ]
    docs = [Document(text=texts[i % len(texts)], title=f"Doc {i + 1}")
            for i in range(n_docs)]
    return Example(qid="q0", question="Which company makes the Civic car model ?",
                   answers=["Honda"], documents=docs)


def make_drqa_example() -> Example:
    """A DRQA-style example: 1 correct, 2 misinfo, 2 temporal, 5 noise documents."""
    spec = [
        ("temporal", "Lee Market reduced waste by 5 percent in Q2 2023 ."),
        ("noise", "The IT helpdesk reported response time under 25 minutes ."),
        ("misinfo", "Lee Market maybe reduced waste by 12 percent in Q2 2024 ."),
        ("correct", "Lee Market reduced waste by 8 percent in Q2 2024 , saving 1 million ."),
        ("noise", "Online customers made recommendations ."),
        ("misinfo", "Approximately 15 percent waste reduction was likely in Q2 2024 ."),
        ("noise", "The store team said revenue growth was current ."),
        ("temporal", "Lee Market reduced waste by 3 percent in Q2 2022 ."),
        ("noise", "Jason Wong coordinated with the IT helpdesk ."),
        ("noise", "The report said average order value increase was 25 percent ."),
    ]
    docs = [Document(text=t, title=f"Doc {i + 1}", doc_type=k)
            for i, (k, t) in enumerate(spec)]
    return Example(qid="0", question="What waste reduction rate did Lee Market achieve in Q2 2024 ?",
                   answers=["Lee Market reduced waste by 8 percent in Q2 2024 ."],
                   documents=docs, wrong_answers=["12 percent", "5 percent"])


def build_tiny_longrope_checkpoint(out_dir: str) -> str:
    """A tiny Phi-3 (long-RoPE) checkpoint whose ``config.json`` has the Hub's
    Phi-3-medium-128k layout: ``original_max_position_embeddings`` at the top
    level and missing from ``rope_scaling`` (transformers>=5 rejects that)."""
    import json

    from transformers import Phi3ForCausalLM

    tok_dir = build_tiny_checkpoint(os.path.join(out_dir, "_tok"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    hub_like = {
        "architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
        "vocab_size": len(tokenizer), "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "max_position_embeddings": 131072, "original_max_position_embeddings": 4096,
        "rope_theta": 10000.0, "torch_dtype": "float32",
        "rope_scaling": {"type": "su", "long_factor": [1.0] * 16, "short_factor": [1.0] * 16},
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": None, "tie_word_embeddings": False,
    }
    patched = json.loads(json.dumps(hub_like))
    patched["rope_scaling"]["original_max_position_embeddings"] = 4096
    torch.manual_seed(0)
    try:            # transformers>=5 wants the RoPE field inside rope_scaling ...
        config = Phi3ForCausalLM.config_class.from_dict(patched)
    except ValueError:  # ... transformers 4.x rejects it there
        config = Phi3ForCausalLM.config_class.from_dict(hub_like)
    model = Phi3ForCausalLM(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hub_like, f, indent=2)
    return out_dir
