# ayn-vqa

Code for the **ImageEval 2026 / ArabicNLP — Task 1a (Spoken VQA), MSA track**
(Codabench `task1a_msa`, competitions #17048/#17001). This repo holds
**pipeline code only**; the dataset itself lives in a separate sibling clone
of the HuggingFace dataset repo `QCRI/AynVQA-ArabicNLP26` (git-LFS, not ours
to push to) and is referenced by path, never copied in here. See
[`../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
for the full research plan and roadmap (M0–M9).

**Current milestone: M2 — ASR bench.** Task 1a (Spoken VQA) only -- no
Task 1b/1c (hallucination detection) work anywhere in this repo. Still no
OCR or VLM calls. `stages/asr.py` has three backends: only
`WhisperLocalASR` (local, offline, no API key) is actually exercised so
far; `FanarAuraASR`/`OpenAITranscribeASR` are complete, ready-to-use
clients gated behind their own API key. No transcript parsing yet -- that
needs an LLM and is deferred alongside M3.

## Layout

```
ayn-vqa-msa/
├── src/ayn_vqa/
│   ├── config.py           # Settings: DATA_ROOT, reports dir, seed, thresholds (env-driven)
│   ├── logging_utils.py    # one place that configures stdlib logging (Rich console handler)
│   ├── artifacts.py        # generic JSONL cache: artifacts/<split>/<stage>/<config_key>.jsonl
│   ├── asr_compare.py      # compare cached ASR runs (e.g. whisper-small vs. whisper-medium)
│   ├── submission.py       # id,prediction CSV writer + vendored format-checker wrapper
│   ├── evaluate.py         # wraps the vendored official scorer (accuracy/bal-acc/macro-F1)
│   ├── experiments.py      # append-only experiments.md log (one row per scored run)
│   ├── run_baseline.py     # orchestrates M1; `aynvqa-predict` CLI entry point
│   ├── run_asr_bench.py    # orchestrates M2; `aynvqa-transcribe` CLI entry point
│   ├── data/
│   │   ├── schema.py       # Task1aRecord: the one Pydantic model every stage agrees on
│   │   ├── loader.py       # JSONL -> validated records, tolerant of malformed rows
│   │   └── validation.py   # does every referenced image/audio file actually exist?
│   ├── stages/
│   │   ├── select.py       # AnswerSelector protocol; RandomSelector, ConstantSelector (M1)
│   │   └── asr.py          # ASRBackend protocol; WhisperLocalASR (run), Fanar/OpenAI (ready)
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
├── docs/{M0_DATA_AUDIT,M1_BASELINES,M2_ASR_BENCH}.md  # design rationale for every module above
├── experiments.md          # append-only: one row per scored run (checked into git)
├── artifacts/              # cached stage outputs, e.g. ASR transcripts (gitignored)
└── reports/                # generated output (gitignored; recreated by the CLIs)
```

## Quickstart

See [`docs/M0_DATA_AUDIT.md`](docs/M0_DATA_AUDIT.md),
[`docs/M1_BASELINES.md`](docs/M1_BASELINES.md), and
[`docs/M2_ASR_BENCH.md`](docs/M2_ASR_BENCH.md) for the full walkthroughs
(design rationale per file, expected output, troubleshooting). Short version:

```bash
uv sync --group dev --group notebook
cp .env.example .env   # then edit AYNVQA_DATA_ROOT if your dataset clone isn't a sibling folder
uv run aynvqa-audit                                          # M0: data audit
uv run aynvqa-predict --selector random --split dev           # M1: trivial baselines
uv run aynvqa-predict --selector constant --constant-value 0 --split dev
uv run aynvqa-transcribe --backend whisper --whisper-model-size small --split dev   # M2: ASR bench
```

Run the test suite with `uv run pytest`.
