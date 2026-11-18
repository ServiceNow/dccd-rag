# Building DRQA from DRBench

This directory regenerates the **DRQA** benchmark from the DRBench source,
bundled under `source/` (text files only: the original PDF/DOCX/PPTX/XLSX renderings
are omitted because the builder reads their Markdown/CSV/JSON equivalents). The
released @10 bundle ships in `../data/DRQA/` and the @5 bundle is derived from it;
use this directory only to regenerate or extend the dataset. LLM generation is not
deterministic, so a rebuild produces a new DRQA variant, not a byte-identical copy.

## Pipeline

For each DRBench *internal insight* (`source/tasks/DR*/files/IN*/qa_dict.json`):

1. **QA pair** — the insight's question/answer becomes a DRQA example.
2. **Conflict documents** are generated around the answer:
   - **correct** — the human-verified DRBench insight file (reused, no LLM).
   - **noise** — DRBench distractor (`DI*`) files (reused, no LLM).
   - **misinfo** — LLM generates a plausible wrong answer with subtle uncertainty
     markers, then writes a format-specific document (email / chat / markdown).
   - **temporal** — LLM generates a historically-accurate value for an older
     `document_date`, written as authoritative for its time.
3. **LLM-as-judge verification** filters each generated document against a fixed
   type-specific rubric (correct entailment / misinfo plausibility & distinctness /
   temporal staleness / noise off-topicness). Rejected examples are dropped.

All generation/verification prompts are the verbatim templates from the paper
(App. A). See `conflict_generator.py` (core + generators) and `build_drqa.py`
(driver + verification rubrics).

## Usage

Requires an `OPENROUTER_API_KEY` (default model `anthropic/claude-opus-4-5`).
Put it in a `.env` file or the environment.

```bash
cd drqa

# Build the @10 bundle (1 correct, 2 misinfo, 2 temporal, 5 noise per example)
python build_drqa.py \
    --num-misinfo-per-answer 2 --num-temporal-per-answer 2 --num-noise 5 \
    --output-dir output/drqa_at10 --save-rejected

# Derive the @5 bundle deterministically from @10
python derive_at5.py --in output/drqa_at10/conflict_dataset.jsonl \
                     --out output/drqa_at5/drqa_at5.jsonl
```

For the @5 composition directly, run with the defaults
(`--num-misinfo-per-answer 1 --num-temporal-per-answer 1 --num-noise 2`).

Useful flags: `--max-examples N` (quick run), `--no-verify` (skip the judge),
`--model <id>` (any OpenRouter model id), `--max-workers N` (parallelism).

## Files

- `build_drqa.py`      — driver: loads DRBench tasks, generates conflict docs, verifies.
- `conflict_generator.py` — `ConflictDatasetGenerator`, document dataclasses, App. A prompts.
- `generators/`        — email / chat / markdown document generators.
- `llm.py`             — OpenRouter / OpenAI client wrapper.
- `utils.py`           — JSONL parsing helpers.
- `derive_at5.py`      — derive the @5 bundle from the @10 bundle.
- `source/`            — DRBench tasks (contexts, personas, insight & distractor files;
  from [ServiceNow/drbench](https://github.com/ServiceNow/drbench), Apache-2.0).
