"""DCCD: Dual-Confidence Contrastive Decoding for Retrieval-Augmented Generation.

Official implementation of the decoding method and baselines from the paper
"Dual-Confidence Contrastive Decoding for Retrieval-Augmented Generation"
(Findings of EACL 2027), with loaders for the DRQA benchmark and the standard
multi-document QA datasets (NQ, TriviaQA, PopQA, RetrievalQA).

Public API
----------
- :class:`DCCD`: answer a question from retrieved documents (high level).
- :func:`generate` / :class:`GenConfig`: decode one example with any mode.
- :func:`load_model_and_tokenizer`: load a model the way the paper runs did.
- :mod:`dccd.data`: dataset loaders (``Example``, ``Document``).
- :mod:`dccd.confidence`: document- and token-level confidence.
- :mod:`dccd.eval`: str-em and the LLM-as-judge for DRQA.
"""

__version__ = "1.0.0"

MODES = ("full", "zeroshot", "cad", "adacad", "cocoa", "dvd", "dccd")
ABLATIONS = ("fixed_gate", "token_only", "doc_only", "random")
ALL_MODES = MODES + ABLATIONS

__all__ = ["DCCD", "GenConfig", "generate", "load_model_and_tokenizer",
           "MODES", "ABLATIONS", "ALL_MODES", "__version__"]


def __getattr__(name):
    # Lazy imports keep `import dccd` (and `dccd.eval`) free of torch/transformers.
    if name == "DCCD":
        from .api import DCCD
        return DCCD
    if name in ("GenConfig", "generate"):
        from . import decoding
        return getattr(decoding, name)
    if name == "load_model_and_tokenizer":
        from .models import load_model_and_tokenizer
        return load_model_and_tokenizer
    raise AttributeError(f"module 'dccd' has no attribute {name!r}")
