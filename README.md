# ayn-vqa

Code for the **ImageEval 2026 / ArabicNLP — Task 1a (Spoken VQA), MSA track**
(Codabench `task1a_msa`, competitions #17048/#17001). This repo holds
**pipeline code only**; the dataset itself lives in a separate sibling clone
of the HuggingFace dataset repo `QCRI/AynVQA-ArabicNLP26` (git-LFS, not ours
to push to) and is referenced by path, never copied in here. See
[`../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
for the full research plan and roadmap (M0–M9).

**Current milestone: M0 — Data Audit.** Nothing beyond loading, validating,
and characterizing the raw data is implemented yet (no ASR, no OCR, no VLM
calls). Later milestones will add `src/ayn_vqa/stages/` (ASR, parsing,
evidence, selection, ensembling) on top of the foundation built here.

## Layout

```
ayn-vqa-msa/
├── src/ayn_vqa/
│   ├── config.py           # Settings: DATA_ROOT, reports dir, seed, thresholds (env-driven)
│   ├── logging_utils.py    # one place that configures stdlib logging (Rich console handler)
│   ├── data/
│   │   ├── schema.py       # Task1aRecord: the one Pydantic model every stage agrees on
│   │   ├── loader.py       # JSONL -> validated records, tolerant of malformed rows
│   │   └── validation.py   # does every referenced image/audio file actually exist?
│   └── audit/
│       ├── audio_stats.py  # duration / sample rate / channels via soundfile (header-only reads)
│       ├── image_stats.py  # resolution / format / mode via Pillow, incl. GIF handling
│       ├── hashing.py      # sha256 (exact dup) + dHash (near dup) over image bytes
│       ├── sampling.py     # seeded random sample + contact-sheet visualization
│       ├── report.py       # aggregates everything into CSV/JSON/Markdown
│       └── run_audit.py    # orchestrates the above; `aynvqa-audit` CLI entry point
├── notebooks/00_data_audit.ipynb
├── tests/                  # fast, offline tests against a synthetic mini-dataset
├── docs/M0_DATA_AUDIT.md   # design rationale for every module above
└── reports/                # generated output (gitignored; recreated by the CLI)
```

## Quickstart

See [`docs/M0_DATA_AUDIT.md`](docs/M0_DATA_AUDIT.md) for the full walkthrough
(design rationale per file, expected output, troubleshooting). Short version:

```bash
uv sync --group dev --group notebook
cp .env.example .env   # then edit AYNVQA_DATA_ROOT if your dataset clone isn't a sibling folder
uv run aynvqa-audit
```

Run the test suite with `uv run pytest`.
