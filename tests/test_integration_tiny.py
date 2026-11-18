"""End-to-end decoding on a tiny, randomly initialised model (CPU, no downloads)."""
from dccd import ALL_MODES
from dccd.decoding import GenConfig, generate
from helpers import make_qa_example


def test_all_modes_run_on_cpu(tiny_model):
    model, tokenizer = tiny_model
    example = make_qa_example(5)
    for mode in ALL_MODES:
        out = generate(model, tokenizer, example,
                       GenConfig(mode=mode, doc_num=5, max_new_tokens=8))
        assert isinstance(out["prediction"], str), mode
        assert out["mode"] == mode


def test_greedy_is_deterministic(tiny_model):
    model, tokenizer = tiny_model
    example = make_qa_example(5)
    cfg = GenConfig(mode="dccd", doc_num=5, max_new_tokens=12, stop_first_sentence=False)
    first = generate(model, tokenizer, example, cfg)["prediction"]
    second = generate(model, tokenizer, example, cfg)["prediction"]
    assert first == second
