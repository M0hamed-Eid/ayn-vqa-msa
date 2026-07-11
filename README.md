# ayn-vqa

Code for the **ImageEval 2026 / ArabicNLP — Task 1a (Spoken VQA), MSA track**
(Codabench `task1a_msa`, competitions #17048/#17001). This repo holds
**pipeline code only**; the dataset itself lives in a separate sibling clone
of the HuggingFace dataset repo `QCRI/AynVQA-ArabicNLP26` (git-LFS, not ours
to push to) and is referenced by path, never copied in here. See
[`../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
for the full research plan and roadmap (M0–M9).

**Current milestone: M1 — repo skeleton + trivial baselines.** Still no
ASR, OCR, or VLM calls anywhere -- `stages/select.py` only has two
model-free baselines (random guess, constant prediction) that exist to
prove the load → predict → submit → score loop end to end. M2 adds the
first real `stages/asr.py`.

## Layout

```
ayn-vqa-msa/
├── src/ayn_vqa/
│   ├── config.py           # Settings: DATA_ROOT, reports dir, seed, thresholds (env-driven)
│   ├── logging_utils.py    # one place that configures stdlib logging (Rich console handler)
│   ├── submission.py       # id,prediction CSV writer + vendored format-checker wrapper
│   ├── evaluate.py         # wraps the vendored official scorer (accuracy/bal-acc/macro-F1)
│   ├── experiments.py      # append-only experiments.md log (one row per scored run)
│   ├── run_baseline.py     # orchestrates the above; `aynvqa-predict` CLI entry point
│   ├── data/
│   │   ├── schema.py       # Task1aRecord: the one Pydantic model every stage agrees on
│   │   ├── loader.py       # JSONL -> validated records, tolerant of malformed rows
│   │   └── validation.py   # does every referenced image/audio file actually exist?
│   ├── stages/
│   │   └── select.py       # AnswerSelector protocol; RandomSelector, ConstantSelector (M1)
│   └── audit/
│       ├── audio_stats.py  # duration / sample rate / channels via soundfile (header-only reads)
│       ├── image_stats.py  # resolution / format / mode via Pillow, incl. GIF handling
│       ├── hashing.py      # sha256 (exact dup) + dHash (near dup) over image bytes
│       ├── sampling.py     # seeded random sample + contact-sheet visualization
│       ├── report.py       # aggregates everything into CSV/JSON/Markdown
│       └── run_audit.py    # orchestrates the above; `aynvqa-audit` CLI entry point
├── scorer_official/        # vendored, unmodified: official scorer + format checker
├── notebooks/00_data_audit.ipynb
├── tests/                  # fast, offline tests against a synthetic mini-dataset
├── docs/{M0_DATA_AUDIT,M1_BASELINES}.md  # design rationale for every module above
├── experiments.md          # append-only: one row per scored run (checked into git)
└── reports/                # generated output (gitignored; recreated by the CLIs)
```

## Quickstart

See [`docs/M0_DATA_AUDIT.md`](docs/M0_DATA_AUDIT.md) and
[`docs/M1_BASELINES.md`](docs/M1_BASELINES.md) for the full walkthroughs
(design rationale per file, expected output, troubleshooting). Short version:

```bash
uv sync --group dev --group notebook
cp .env.example .env   # then edit AYNVQA_DATA_ROOT if your dataset clone isn't a sibling folder
uv run aynvqa-audit                                        # M0: data audit
uv run aynvqa-predict --selector random --split dev         # M1: trivial baselines
uv run aynvqa-predict --selector constant --constant-value 0 --split dev
```

Run the test suite with `uv run pytest`.
