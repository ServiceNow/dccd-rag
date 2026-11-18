# Datasets

Only **DRQA**, the paper's benchmark, ships with this repository. The four standard
multi-document QA datasets are large and separately licensed, so they are fetched from
their original sources:

```bash
python scripts/verify_data.py --download                                  # PopQA + RetrievalQA (Hugging Face)
python scripts/verify_data.py --download --fid_dir <FiD>/open_domain_data  # + NQ + TriviaQA
```

`verify_data.py` downloads what it can, derives DRQA@5, and checks every dataset's
example count and schema (PASS / SKIP / FAIL). `scripts/download_data.py <dataset>`
fetches a single dataset.

| Dataset | File | # Examples | Metric | How to get it |
|---|---|---:|---|---|
| **DRQA** @10 | `DRQA/drqa_at10.jsonl` | 568 | LLM-as-judge | bundled |
| **DRQA** @5 | `DRQA/drqa_at5.jsonl` | 568 | LLM-as-judge | derived from @10 (`python drqa/derive_at5.py`) |
| NQ | `NQ/test.json` | 3,610 | str-em | FiD preprocessing |
| TriviaQA | `TQA/test.json` | 11,313 | str-em | FiD preprocessing |
| PopQA | `PopQA/test.json` | 4,267 | str-em | Hugging Face |
| RetrievalQA | `RetrievalQA/eval_all.json` | 2,785 | str-em | Hugging Face |

## DRQA

DRQA turns the 568 synthetic internal facts of [DRBench](https://github.com/ServiceNow/drbench)
into questions. Each question is paired with a bundle of enterprise documents (Markdown
reports, e-mail threads and chat logs): one **correct** document holding the current
answer, plus **misinformation** documents that state a plausible wrong value,
**temporal** documents that state a value that was true at an earlier date, and
**noise** documents that are on topic but do not answer the question.

| Documents per example | correct | misinfo | temporal | noise | total |
|---|---:|---:|---:|---:|---:|
| @5 | 1 | 1 | 1 | 2 | 5 |
| @10 | 1 | 2 | 2 | 5 | 10 |

At @10, four examples have fewer than five noise documents (3 examples have 8 documents
and 1 has 7), because their DRBench tasks have too few distinct distractors.

### What the model sees

Each document reaches the model as `Document [i](Title: Doc i) <text>`, where
`<text>` is the document body. E-mail and chat documents keep DRBench's JSONL format
(one message per line); the first line of an e-mail document is the persona's synthetic
mailbox account record, as in DRBench. All other fields are **metadata, never shown to the
model**: the document type, the stated answer, the source format, the `document_date`,
and the confidence score.

The `document_date` is never prepended to the input. Generated distractors often
mention their own date in their body (an e-mail header, a "Date:" line, a chat
timestamp), but the correct documents never do. Counting documents whose own date appears
in the body in a common date format:

| Document type | Date appears in body |
|---|---|
| temporal | at least 786 / 1,136 |
| misinformation | 441 / 1,136 |
| noise | 5 / 2,831 |
| correct | 0 / 568 |

### DRQA@5

`drqa_at5.jsonl` is a type-stratified subset of `drqa_at10.jsonl`, written by
`drqa/derive_at5.py`. For each example it keeps the first correct, the first
misinformation, the first temporal and the first two noise documents, then shuffles
them with a fixed per-example seed (42 + line index). The original @5 bundle used for
the paper's @5 runs was not preserved. This subset has the composition the paper
describes (1 / 1 / 1 / 2), but DRQA@5 numbers can differ slightly from the paper's
Table 1. DRQA@10 is the exact bundle used in the paper.

### Schema

JSONL, one example per line (RAMDocs-style):

```json
{"question": "...",
 "documents": [{"text": "...",
                "type": "correct | misinfo | temporal | noise",
                "answer": "the value this document states",
                "format": "md | email | chat",
                "document_date": "YYYY-MM-DD",
                "confidence": 0.0,
                "reliability_signals": {"recency": "...", "verification": "...", "authority": "..."},
                "source_fact": {"...": "DRBench fact this document was built from"},
                "file_path": "tasks/DR0002/files/DI002_xlsx"}],
 "disambig_entity": ["Lee's Market"],
 "gold_answers": ["..."], "wrong_answers": ["...", "..."],
 "gold_answers_full": ["..."], "wrong_answers_full": ["..."]}
```

- `confidence` is a synthetic score drawn from a range that depends on the
  document type. It reveals the type, so it must not be used as a model input.
- `file_path` is the DRBench fact directory the document came from, relative to
  `drqa/source/`. It is `null` for LLM-generated documents (misinformation and temporal).

```python
from dccd.data import load_dataset
examples = load_dataset("data/DRQA/drqa_at10.jsonl", "drqa")
doc = examples[0].documents[0]
doc.text, doc.doc_type, doc.document_date, doc.doc_format
```

To rebuild DRQA from the DRBench source, see [`drqa/README.md`](../drqa/README.md).

DRQA is for research use only. It contains deliberately generated misinformation and
outdated evidence and must not be used as a factual source. All companies, people and
facts in it are synthetic.

## Standard QA datasets

All four use the same JSON schema; the decoder reads `ctxs[:doc_num]` to get @5 or @10:

```json
[{"question": "...", "answers": ["...", "..."],
  "ctxs": [{"id": "...", "title": "...", "text": "..."}, ...]}]
```

### NQ: Natural Questions (3,610 examples; 100 ctxs each)
Kwiatkowski et al. (2019), with the DPR retrieval and Fusion-in-Decoder reader data of
Izacard and Grave (2021). Run the FiD data pipeline first, then convert:
```bash
git clone https://github.com/facebookresearch/FiD && (cd FiD && bash get-data.sh)
python scripts/download_data.py nq --fid_file FiD/open_domain_data/NQ/test.json
```

### TriviaQA (11,313 examples; 100 ctxs each)
Joshi et al. (2017), with the same FiD/DPR preprocessing as NQ:
```bash
python scripts/download_data.py tqa --fid_file FiD/open_domain_data/TQA/test.json
```

### PopQA (4,267 examples; 10 ctxs each)
Mallen et al. (2023). We use the retrieval-augmented release
[`Atipico1/popQA_preprocessed`](https://huggingface.co/datasets/Atipico1/popQA_preprocessed) (`test` split):
```bash
python scripts/download_data.py popqa
```

### RetrievalQA (2,785 examples; 10 ctxs each)
Zhang et al. (2024), [`aialt/RetrievalQA`](https://huggingface.co/datasets/aialt/RetrievalQA),
converted to the schema above:
```bash
python scripts/download_data.py retrievalqa
```
