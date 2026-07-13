# AynVQA-MSA

Pipeline code for **ImageEval 2026 / ArabicNLP — Task 1a (Spoken Visual Question Answering), MSA track** (Codabench `task1a_msa`, competition #17001; the sibling English track `task1a_en` is #17002). Given a spoken Arabic (Modern Standard Arabic) audio question about an image, plus that image, the system must transcribe the question, recover its three multiple-choice options, and select the correct option — a 3-way classification problem end to end from raw audio.

This repository holds **pipeline code only**. The dataset itself (`QCRI/AynVQA-ArabicNLP26`, git-LFS) lives in a separate sibling clone and is referenced by path via configuration, never copied into this repo.

**Current status: Milestone M6 complete and frozen.** The production pipeline scores **79.00% accuracy (395/500)** on the full `dev`/`msa` split — up from an M4 baseline of 74.80% (374/500), and dramatically ahead of the official Qwen2.5-Omni baseline (39.8% on MSA). This README documents the project as it stands after M6: what it does, how it's built, why every major design decision was made, and how to run it yourself.

---

## 1. Project overview

### What AynVQA is

AynVQA-MSA tackles **Task 1a** of the ImageEval 2026 shared task: spoken visual question answering in Modern Standard Arabic. Each record in the dataset pairs:

- an **image** (a photograph depicting some object, scene, place, or activity),
- a **spoken audio clip** (a question about the image, asked in MSA, that also verbally states three candidate answers), and
- a **gold label** (`0`, `1`, or `2` — which of the three spoken options is correct), present for `train`/`dev` but withheld for the blind `devtest`/`test` splits.

The system must listen to the audio, understand both the question and the three options it contains, look at the image, and pick the correct option index. There is no separate text channel for the question or options — everything the system knows about what's being asked comes from transcribing the spoken audio itself.

### The problem, concretely

This is harder than a typical VQA benchmark for three compounding reasons:

1. **The question and its three options only exist as speech.** Before any visual reasoning can happen, the pipeline must first accurately transcribe Arabic speech, then correctly *segment* that transcript into a question plus exactly three distinct options — a structured-extraction problem with its own failure modes (fabricated options, boilerplate leaking into an option, options in the wrong order, or fewer than three genuine options).
2. **MSA transcription is meaningfully harder than English speech in this dataset.** The official baseline itself reflects this: 39.8% accuracy on the MSA track vs. 66.4% on the English track for the same underlying task design.
3. **The dataset's own official baseline parser is fragile.** The organizers' reference implementation uses a bare regex (`re.search(r"[012]", ...)`) to pull an answer index out of free-text model output, silently falling back to index `0` on any parse failure — a design this project deliberately avoids (see §2, Select stage).

### The dataset

Task 1a data lives in `task1a/<split>_<language>.jsonl`, one JSON object per line, validated on load against a single shared schema (`Task1aRecord`, `frozen=True`, `extra="forbid"` — any unexpected key fails loudly rather than being silently dropped). Splits are `train`, `dev`, `devtest` (the blind `test` split does not exist on disk yet, expected 2026-07-20). Languages are `en` (present for internal audit/comparison only — not a valid scored input for this competition track) and `msa` (the competition track, and this project's default everywhere). `country`, `category`, `subcategory`, and `label` are present on `train`/`dev` but absent (not merely `null`) on `devtest`, matching the organizers' own blind-split convention.

### The pipeline, in one sentence

**ASR (Whisper) → Parse (Ollama VLM, text-only) → Repair escalation (conditional) → Select (Ollama VLM, joint-MCQ) → submission CSV → format check → scoring → error analysis.**

Every stage's output is cached to disk, so re-running after a crash, an interruption, or a config tweak only recomputes what actually changed (see §9, Caching system).

### Why it's designed this way

- **Local, open-weight VLMs served by Ollama, not a hosted API.** No per-call cost, no rate limits, no data leaving the machine, and (once an idle GPU was found) fast enough for full-split runs. This shaped almost every later decision — including which VLM was and wasn't a candidate for the M6 swap (see §5, §17).
- **A cascade of small, independently-cacheable stages rather than one monolithic call.** ASR, parsing, and selection are conceptually different jobs, fail in different ways, and benefit from being independently ablatable (see §9, §16).
- **Schema-constrained JSON decoding everywhere a model output must be machine-parsed** (parse stage: `{question, option_0, option_1, option_2}`; select stage: `answer_index` constrained to `{0,1,2}`) — explicitly designed from M3 onward to avoid the official baseline's brittle regex-and-silent-fallback failure mode.
- **Vendored, unmodified official scorer and format checker** (`scorer_official/`) so every reported number is directly comparable to what the competition leaderboard itself would compute.
- **Every change validated at full scale before being adopted** — a discipline that recurs throughout the milestone history (§10, §12) and is the reason two fully-implemented, tested features (chain-of-thought prompting and few-shot exemplar retrieval) sit behind off-by-default flags today despite being complete.

### Current production pipeline (as of M6)

| Stage | Backend | Notes |
|---|---|---|
| ASR (primary) | Whisper `medium`, local, CPU/int8 | faster-whisper/CTranslate2 |
| ASR (repair escalation) | Whisper `large-v3`, local, GPU/float16 | only invoked when repair triggers |
| Parse | Ollama VLM, **text-only**, prompt **v2** | `qwen2.5vl:7b` |
| Repair | conditional 3-step escalation ladder | **enabled by default** |
| Select | Ollama VLM, joint-MCQ (image + all 3 options in one prompt) | `qwen3-vl:8b` (changed from `qwen2.5vl:7b` in M6) |
| Chain-of-thought | implemented, tested | **off by default** (`cot_enabled=False`) |
| Few-shot retrieval | implemented, tested | **off by default** (`fewshot_enabled=False`) |

### Current default models

- `AYNVQA_OLLAMA_PARSE_MODEL` → `qwen2.5vl:7b` (unchanged since M3 — only the select stage was evaluated for a swap)
- `AYNVQA_OLLAMA_SELECT_MODEL` → `qwen3-vl:8b` (changed from `qwen2.5vl:7b` in M6, +4.2 points)
- `AYNVQA_WHISPER_MODEL_SIZE` → `medium` (primary ASR pass)
- `AYNVQA_REPAIR_WHISPER_MODEL_SIZE` → `large-v3` (repair-escalation ASR pass, GPU only)

---

## 2. High-level architecture

```
                 ┌───────────────────────────────────────────────────────────────┐
                 │                         INPUT RECORD                          │
                 │        image.jpg  +  audio.wav  +  (gold label, if any)       │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  ASR STAGE            stages/asr.py — WhisperLocalASR         │
                 │  audio.wav -> raw transcript text (Whisper "medium", CPU)     │
                 │  cache: artifacts/<split>/asr/<lang>_whisper-medium.jsonl     │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │ transcript text
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  PARSE STAGE          stages/parse.py — OllamaTranscriptParser│
                 │  transcript text -> {question, option_0, option_1, option_2}  │
                 │  text-only VLM call, JSON-schema constrained decoding, v2     │
                 │  cache: artifacts/<split>/parse/<lang>_ollama-parse.jsonl     │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │ parsed question + 3 options
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  REPAIR STAGE (conditional, M4)     run_pipeline.py           │
                 │  option_quality.check_option_quality() flags degenerate parse │
                 │    step 1: stricter same-transcript reparse                   │
                 │    step 2: escalate to Whisper large-v3 (GPU) + reparse       │
                 │    step 3: give up, proceed unrepaired                        │
                 │  only runs if repair_enabled=True (default)                   │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │ (possibly repaired) question + 3 options
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  VLM SELECTOR STAGE   stages/select_vlm.py                    │
                 │  OllamaJointMCQSelector: image + question + 3 options in ONE  │
                 │  prompt -> answer_index constrained to {0,1,2}                │
                 │  model: qwen3-vl:8b (default since M6)                        │
                 │  optional: CoT reasoning field, few-shot exemplars (both off) │
                 │  cache: artifacts/<split>/select/<lang>_<selector.name>.jsonl │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │ predicted index {0,1,2}
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  PREDICTION            submission.py                         │
                 │  write_predictions_csv -> prediction_<config_key>_<split>_    │
                 │  <language>.csv  (columns: id, prediction)                    │
                 └───────────────────────────────┬───────────────────────────────┘
                                                   │
                                                   ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │  EVALUATION            submission.py + evaluate.py            │
                 │  1. format check   -> vendored check_1a (scorer_official/)    │
                 │  2. scoring         -> vendored score_1a (accuracy/           │
                 │     balanced_accuracy/macro_f1), logged to experiments.md     │
                 │  3. error analysis -> error_analysis.py: per-record CSV,      │
                 │     per-country/category breakdowns, pipeline-artifact split  │
                 └───────────────────────────────────────────────────────────────┘
```

### Stage-by-stage detail

**1. Load split.** `data/loader.py`'s `load_split(data_root, split, language)` reads `task1a/<split>_<language>.jsonl` line by line — not via `pandas.read_json` or `datasets.load_dataset`, both of which abort on the first malformed line. Bad lines become `RecordError`s (with a line number) instead of crashing the whole load. Optionally seeded-subsampled via `sample_records(records, sample_n, seed)`, using `random.Random(seed).sample`, never a plain slice — the JSONL is sorted alphabetically by country, so a slice would sample one country only.

**2. ASR stage** (`stages/asr.py`, `_run_asr_stage` in `run_pipeline.py`). Each audio file is transcribed by an `ASRBackend`. Three backends exist behind one Protocol: `WhisperLocalASR` (faster-whisper, local, the only one actually run in production), `FanarAuraASR` (QCRI Fanar Aura-STT API), and `OpenAITranscribeASR` (`gpt-4o-transcribe`) — the latter two are complete, unit-tested via `httpx.MockTransport`, but never exercised against a live API key ("ready-but-unexercised"). Every result — including a failed `Transcript` with `error` set — is written to cache immediately, so transient failures aren't silently retried forever but also aren't retried at all without clearing the cache.

**3. Parse stage** (`stages/parse.py`, `_run_parse_stage`). A **text-only** call to a local Ollama VLM (`OllamaTranscriptParser`) turns the raw transcript into a structured `{question, option_0, option_1, option_2}` object via JSON-schema-constrained decoding. If the upstream transcript failed or is empty, this stage short-circuits with a synthetic failed parse without ever touching the cache — so once the ASR problem is fixed, parse will naturally retry on the next run with no manual cache surgery. The prompt has gone through two revisions; **prompt v2** is production (see §9 for why the v1→v2 swap required manual cache handling).

**4. Repair stage** (conditional, M4; `_run_repair_stage` in `run_pipeline.py`). Runs only when `repair_enabled=True` (the default) **and** the parser is an `OllamaTranscriptParser`. Any parse flagged as structurally-valid-but-semantically-degenerate by `option_quality.check_option_quality()` (empty/too-short options, leftover Arabic boilerplate like "الخيارات هي", duplicate options, a fabricated placeholder, one option being a substring of another) enters a 3-step escalation ladder: (1) a stricter same-transcript reparse; (2) if still degenerate, re-transcribe with Whisper `large-v3` on GPU and reparse; (3) if still degenerate, give up and proceed unrepaired. Each step writes to its own isolated cache namespace (see §9).

**5. Retrieval stage** (conditional, M5; `_run_retrieval_stage`). Only runs when `fewshot_enabled=True` (off by default). For each query image, `OllamaCategoryRetriever` classifies it into a fixed, closed 9-category/31-subcategory taxonomy via a JSON-schema-enum-constrained VLM call, then seeded-randomly samples up to `k` matching records from `train`, hydrating each into a full `Exemplar` (question/options/gold answer) by running the *same* ASR+parse stage functions against `train`'s own cache namespace.

**6. Select stage** (`stages/select_vlm.py`, `_run_select_stage`). `OllamaJointMCQSelector` shows the image and all three options together **in one prompt** (deliberately, to neutralize the failure mode of asking about each option in isolation and to match how the official baseline is scored), and asks for `answer_index` constrained by JSON schema to `{0,1,2}` — no free-text regex parsing anywhere in this path. If parsing upstream failed or produced no options, this stage returns a fallback prediction (`FALLBACK_INDEX`) directly, bypassing the cache — cheap enough to redo every run, and self-healing once parsing succeeds. Optionally augmented with a chain-of-thought schema (`visible_details` + `reasoning` fields before `answer_index`) or multi-turn few-shot exemplar conversation — both implemented, both off by default.

**7. Write submission CSV.** `write_predictions_csv` writes exactly `id,prediction` columns to `prediction_<config_key>_<split>_<language>.csv`, where `config_key` combines the ASR, parser, and (possibly repair-/few-shot-suffixed) selector cache keys.

**8. Format check.** `check_submission_format` runs the vendored, unmodified official `check_1a` checker (`scorer_official/format_checker/check_format.py`) against the freshly-written CSV. The process exits with `SystemExit(1)` if this fails.

**9. Scoring** (conditional on gold labels being present — skipped for `devtest`/`test`). `score_predictions` wraps the vendored, unmodified official `score_1a` (`scorer_official/scorer/score.py`), producing `accuracy` / `balanced_accuracy` / `macro_f1`, and appends one row to the git-tracked `experiments.md` via `experiments.py`'s `log_experiment`.

**10. Error analysis.** `error_analysis.build_error_table` joins gold record, transcript, parse, and prediction into one `pandas.DataFrame` per record (including `option_quality_ok`/`option_quality_reasons` and `asr_repetition_loop` columns), written to `error_analysis_<config_key>_<split>_<language>.csv`, plus console breakdowns by country/category and a pipeline-artifact-vs-genuine-miss split via `summarize_option_quality`.

Producing `prediction.csv` is as far as this pipeline goes — **uploading to Codabench remains a manual step.**

---

## 3. Repository structure

```
ayn-vqa-msa/
├── .env                     # gitignored; your real local config
├── .env.example              # committed template documenting AYNVQA_* vars (partially stale, see §6)
├── .python-version           # "3.13"
├── README.md
├── experiments.md            # append-only, git-tracked: one row per scored run
├── pyproject.toml            # deps, CLI entry points, ruff/mypy/pytest config
├── uv.lock
├── src/ayn_vqa/               # the package — see full breakdown below
├── tests/                     # 23 test files + conftest.py, 2,278 lines, fully offline
├── docs/                      # 7 milestone design docs, M0-M6
├── notebooks/                  # 00_data_audit.ipynb (companion to the M0 audit CLI)
├── scorer_official/            # vendored, unmodified: official scorer + format checker
├── reports/                     # generated output (gitignored except .gitkeep)
├── artifacts/                    # cached stage outputs (gitignored except .gitkeep)
└── .venv/, .mypy_cache/, .ruff_cache/, .pytest_cache/   # tool caches (gitignored)
```

### `src/ayn_vqa/` — the package

```
ayn_vqa/
├── config.py             # Settings: every path/seed/model/flag, env-driven (AYNVQA_* prefix)
├── logging_utils.py      # configures stdlib logging with a Rich console handler (+ Windows UTF-8 fix)
├── artifacts.py           # generic JSONL cache: artifact_path / read_jsonl_cache / append_jsonl
├── ollama_client.py       # thin httpx wrapper for local Ollama /api/chat (single- and multi-turn, image, JSON schema)
├── option_quality.py      # rule-based detector for semantically-degenerate (but schema-valid) parsed options
├── error_analysis.py      # pure-function join of transcript+parse+prediction+gold into one error table
├── evaluate.py             # wraps the vendored official scorer; turns its sys.exit into RuntimeError
├── submission.py           # writes id,prediction CSV + runs the vendored official format checker
├── experiments.py          # append-only, git-tracked experiment log (experiments.md)
├── asr_compare.py          # read-only comparison of two+ cached ASR runs (no model calls, no CLI)
├── run_baseline.py         # CLI (`aynvqa-predict`): M1 trivial baselines
├── run_asr_bench.py        # CLI (`aynvqa-transcribe`): M2 ASR-only bench
├── run_pipeline.py         # CLI (`aynvqa-run`): the full M3+ cascade
├── data/
│   ├── schema.py            # Task1aRecord (pydantic, frozen, extra="forbid"); Language/Split enums
│   ├── loader.py             # tolerant line-by-line JSONL loader (bad lines -> RecordError, never crashes)
│   └── validation.py         # checks every image/audio path referenced by a record actually exists
├── audit/
│   ├── run_audit.py          # CLI (`aynvqa-audit`): M0 orchestrator
│   ├── audio_stats.py         # fast WAV header probing (soundfile.info) for duration/rate/channels/subtype
│   ├── image_stats.py         # fast image header probing (PIL lazy open) for resolution/format/mode
│   ├── hashing.py              # exact (SHA-256) and near-duplicate (dHash) image detection
│   ├── sampling.py             # seeded random sampling (never a slice) + contact-sheet renderer
│   └── report.py               # aggregates audit dataclasses into CSV/JSON/Markdown + PNG figures
└── stages/
    ├── asr.py                  # ASRBackend Protocol + WhisperLocalASR / FanarAuraASR / OpenAITranscribeASR
    ├── parse.py                 # TranscriptParser Protocol + OllamaTranscriptParser
    ├── retrieve.py               # ExemplarRetriever Protocol + OllamaCategoryRetriever (M5 few-shot)
    ├── select.py                  # AnswerSelector Protocol + RandomSelector/ConstantSelector (M1 baselines)
    └── select_vlm.py               # VLMSelector Protocol + OllamaJointMCQSelector (joint-MCQ, optional CoT/few-shot)
```

### `tests/`

23 test modules + `conftest.py` (2,278 lines total). Fully offline: no network calls, no real model weights, no real multi-gigabyte dataset. `conftest.py` defines the shared `mini_dataset(tmp_path)` fixture — a tiny synthetic dataset built entirely in code (5 random-noise JPEGs + 1 byte-identical duplicate, one WAV per record with varying duration/channels, a 7-row `train_msa.jsonl` with one intentionally malformed line, and minimal `dev_msa.jsonl`/`devtest_msa.jsonl`). Every test runs against this fixture. Notable files: `test_run_pipeline.py` (447 lines, the largest — exercises the full M3+ cascade with fake stages, calling `run_pipeline()` directly with explicit kwargs like `repair_enabled=True`, `fewshot_enabled=True`), `test_select_vlm.py`, `test_option_quality.py`, `test_retrieve.py`, `test_ollama_client.py` (request-shape verification via `httpx.MockTransport`, no live server).

### `docs/`

Seven milestone design documents (1,592 lines total), `M0_DATA_AUDIT.md` through `M6_VLM_SWAP.md` — see §14 for the full index with one-line descriptions of each.

### `reports/`

Generated output, gitignored except `.gitkeep`. Populated per-run with subfolders named after the CLI/milestone that produced them (e.g. `m0_data_audit/`, `m1_baselines/`, `m3_pipeline/`, `m4_full_ablation/`, `m4_parser_v2/`, `m5_cot/`, `m5_fewshot/`, `m6_qwen3vl/`), each holding `prediction_*.csv` and/or `error_analysis_*.csv` files named after their config key.

### `artifacts/`

The JSONL cache layout described in §9:
```
artifacts/
├── _v1_prompt_archive/{parse,select}/   # manually archived pre-v2-prompt cache (see §9)
├── dev/{asr,parse,select}/
└── train/{asr,parse}/
```

### `scorer_official/`

Vendored, unmodified upstream code from the competition organizers:
```
scorer_official/
├── format_checker/check_format.py   # check_1a
└── scorer/{backbone.py, score.py}    # score_1a
```
Explicitly excluded from ruff linting (`extend-exclude` in `pyproject.toml`) — "not ours to reformat or fix lint findings in." Every accuracy number this project reports comes from calling this code directly, unmodified, so it is comparable to the actual leaderboard.

### Other top-level items

- `notebooks/00_data_audit.ipynb` — interactive companion to the M0 audit CLI; requires the `notebook` dependency group.
- `experiments.md` — append-only, auto-generated (via `experiments.py`'s `log_experiment`) table of every scored run: selector/split/language/accuracy/balanced_accuracy/macro_f1 with UTC timestamps. This is the ground-truth numeric log every milestone doc's headline numbers are cross-checked against.
- **No `.github/` directory and no CI configuration exists anywhere in this repo.** Testing/linting/type-checking are run manually (see §15).

---

## 4. Installation

### Prerequisites

- **Python 3.13+** (`.python-version` pins `3.13`).
- **[uv](https://docs.astral.sh/uv/)** — the project uses `uv`'s own build backend (`uv_build`), not setuptools/hatchling, and `uv.lock` is checked in.
- **[Ollama](https://ollama.com)**, installed and reachable locally — required for the parse and select stages (M3+); not required for M0/M1/M2.
- Optionally, an **NVIDIA GPU** — only needed for GPU-accelerated Whisper (the M4 repair-escalation ASR step defaults to `cuda`).

### Steps

**1. Clone this repository:**
```bash
git clone <repo-url> ayn-vqa-msa
cd ayn-vqa-msa
```

**2. Clone the dataset as a sibling directory.** `AYNVQA_DATA_ROOT` defaults to `../AynVQA-ArabicNLP26`, so by default the dataset repo is expected to live one directory level up from `ayn-vqa-msa/`:
```bash
cd ..
git clone https://huggingface.co/datasets/QCRI/AynVQA-ArabicNLP26
cd ayn-vqa-msa
```
This is a git-LFS repository — make sure `git lfs pull` has actually materialized the image/audio files (see §18 for a real incident where 3 images were left as unmaterialized LFS pointer stubs). If your dataset clone lives elsewhere, set `AYNVQA_DATA_ROOT` accordingly instead (see §6).

**3. Install dependencies with uv:**
```bash
uv sync --group dev --group notebook
```
- **Base only** (`uv sync`, no groups): everything needed to run the pipeline on CPU — Whisper on CPU/int8, Ollama-based parsing/selection (Ollama is an external local server, not a Python dependency), data audit, baselines, submission/scoring.
- **`--group dev`**: linting (`ruff`), type-checking (`mypy`, plus `numpy`/`pandas-stubs` for accurate stub coverage), and testing (`pytest`, `pytest-cov`). Install this for any development work.
- **`--group gpu`**: `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` — the CUDA runtime DLLs `ctranslate2` (faster-whisper's inference backend) needs at runtime. **Deliberately kept out of core dependencies** so installing on a machine without an NVIDIA GPU (CPU-only laptop, CI, non-NVIDIA hardware) doesn't force multi-gigabyte CUDA wheel downloads that would never be used. Add it only if you intend to run Whisper with `AYNVQA_WHISPER_DEVICE=cuda`.
- **`--group notebook`**: `ipykernel` + `jupyterlab`, needed only to open `notebooks/00_data_audit.ipynb`.
- Groups combine freely: `uv sync --group dev --group gpu --group notebook` installs everything.

**4. Set up your `.env`:**
```bash
cp .env.example .env
```
Then edit `AYNVQA_DATA_ROOT` if your dataset clone isn't a sibling folder, and optionally fill in `AYNVQA_FANAR_API_KEY`/`AYNVQA_OPENAI_API_KEY` if you want to exercise the (currently unused-in-production) Fanar or OpenAI ASR backends. **Note:** `.env.example` is missing several sections present in `config.py` (the entire M4/M5 groups) — see §6 for the complete, accurate field list.

**5. Ollama setup** (only required before M3+, i.e. before `aynvqa-run`) — see §5 below. Not needed for M0 (`aynvqa-audit`), M1 (`aynvqa-predict`), or M2 (`aynvqa-transcribe`), which run without any local model server.

**6. Verify the install:**
```bash
uv run pytest
```
Fully offline — uses the synthetic `mini_dataset` fixture, doesn't touch the real dataset, doesn't require Ollama or a GPU.

### GPU dependencies, explained

Local Whisper transcription runs through `faster-whisper`, whose actual inference backend is `ctranslate2`. On a CUDA-capable machine you can set `AYNVQA_WHISPER_DEVICE=cuda` (and typically `AYNVQA_WHISPER_COMPUTE_TYPE=float16`) to run on GPU — this requires the `gpu` dependency group installed (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). The production repair-escalation ASR step (`AYNVQA_REPAIR_WHISPER_DEVICE=cuda`, `large-v3`, `float16` by default) assumes this GPU path is available.

### Whisper dependencies

`faster-whisper` is a core dependency (always installed) — CPU inference (the primary ASR pass default: `medium`/`cpu`/`int8`) works out of the box with no extra setup, no API key, and no GPU.

### Ollama requirement

Ollama is an **external local server**, not a Python package — it must be installed separately (see §5) and running (`ollama serve`) before `aynvqa-run` (M3+) will succeed. Without it, the pipeline fails with connection errors against `AYNVQA_OLLAMA_BASE_URL` (default `http://localhost:11434`).

### Windows-specific notes

- **CUDA DLL registration.** `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` ship their DLLs inside the pip wheel rather than relying on a system-wide CUDA Toolkit install. PyTorch registers its own copies of these DLL directories at import time, but `ctranslate2` does not — so on Windows, `device="cuda"` would otherwise fail (not at model construction, but at the *first* `.transcribe()` call) with an error like `"cublas64_12.dll is not found"`, even with the packages correctly installed and a working GPU present. This repo works around it automatically in code (`stages/asr.py`'s `_register_windows_cuda_dll_dirs()`, gated to `device.startswith("cuda")` and `sys.platform == "win32"`) — **no manual install step is required**; this is handled transparently. On Linux this function is a no-op, since the dynamic linker resolves these libraries via the wheel's own rpath directly.
- **Console Unicode.** Arabic transcript/console output on Windows previously crashed with `UnicodeEncodeError` because Rich's legacy console renderer wrote through the system ANSI codepage (`cp1252`), which cannot encode Arabic. `logging_utils.py` forces `legacy_windows=False` and reconfigures `stdout`/`stderr` to UTF-8 — again, handled automatically, no action needed from you.
- No other Windows-specific PATH or install steps exist in this project.

---

## 5. Ollama setup

Ollama is required starting at M3 (i.e., for `aynvqa-run`). It is not required for `aynvqa-audit`, `aynvqa-predict`, or `aynvqa-transcribe`.

### Install Ollama

Follow the instructions at [ollama.com](https://ollama.com) for your platform, then start the server:
```bash
ollama serve
```

### Pull the models this project actually uses today

```bash
ollama pull qwen2.5vl:7b     # parse stage (text-only transcript structuring) -- also the pre-M6 select-stage default
ollama pull qwen3-vl:8b      # select stage (joint-MCQ image answering) -- current production default since M6
```

| Stage | Model | Size | Why |
|---|---|---|---|
| **Parse** (`AYNVQA_OLLAMA_PARSE_MODEL`) | `qwen2.5vl:7b` | ~5.97GB | Unchanged since M3. This stage was never separately re-evaluated during the M6 model swap — only the select stage was — so it stays on the model that has always worked here. |
| **Select** (`AYNVQA_OLLAMA_SELECT_MODEL`) | `qwen3-vl:8b` | ~6.14GB | Changed in M6 from `qwen2.5vl:7b`. A full 500-item, otherwise-identical-pipeline ablation showed +4.2 points (79.00% vs 74.80%) — the largest single-change accuracy gain of the whole project (see §10, §13). |

### Why the parse and select models differ

The two stages started on the same model (`qwen2.5vl:7b`) through M5. In M6, only the **select** stage — the one doing actual visual reasoning over the image — was evaluated for a model swap; the **parse** stage (text-only transcript structuring) was left untouched because it was not separately benchmarked. This is a deliberate, documented scope limitation, not an oversight: `docs/M6_VLM_SWAP.md` explicitly frames the parser-model question as unevaluated future work.

### A model that was considered and rejected: InternVL

M6 directly checked InternVL against Ollama's official model catalog and found it **not available** — only unofficial community GGUF conversions exist (e.g. third-party `blaifa/InternVL3` uploads), with no first-party quality assurance and unverified compatibility with the JSON-schema-constrained decoding this entire pipeline depends on for every single prediction. InternVL was ruled out on that basis; Qwen3-VL (a first-party official Ollama model, and the direct successor line to the already-used Qwen2.5-VL) was selected instead. See §18 for more on this.

---

## 6. Configuration

All configuration lives in `src/ayn_vqa/config.py`'s `Settings` class (`pydantic-settings`), loaded from `PROJECT_ROOT/.env` with prefix `AYNVQA_` — every field `foo_bar` is overridable by env var `AYNVQA_FOO_BAR` or a matching line in `.env`. `PROJECT_ROOT` is computed once as two parents up from `config.py`, so path resolution behaves identically regardless of your current working directory (invoking from the project root, a notebook, or a test runner started elsewhere all resolve the same way).

Relative `data_root`/`reports_dir`/`artifacts_dir` values resolve against `PROJECT_ROOT`, not `Path.cwd()`. Absolute paths pass through unchanged.

**Important:** `.env.example` (the committed template) is missing the entire M4 (repair) and M5 (CoT/few-shot) sections below, and its `AYNVQA_OLLAMA_SELECT_MODEL` line is stale (still shows the pre-M6 `qwen2.5vl:7b` default). Copying `.env.example` verbatim into `.env` will not surface these 8 fields, and would silently downgrade your select model if you uncomment that stale line. The table below is the authoritative, current source of truth — always cross-check against `config.py` itself.

### General / paths

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `data_root` | `../AynVQA-ArabicNLP26` | `AYNVQA_DATA_ROOT` | Path to the dataset clone; sibling-directory convention. |
| `reports_dir` | `reports` | `AYNVQA_REPORTS_DIR` | Where reports/predictions are written. |
| `artifacts_dir` | `artifacts` | `AYNVQA_ARTIFACTS_DIR` | Where cached stage outputs are written. |
| `random_seed` | `42` | `AYNVQA_RANDOM_SEED` | Seed for all deterministic sampling. |
| `sample_grid_n` | `24` | `AYNVQA_SAMPLE_GRID_N` | Items drawn for the audit contact-sheet grid. |
| `near_dup_max_distance` | `4` | `AYNVQA_NEAR_DUP_MAX_DISTANCE` | Max Hamming distance (of 64 bits) for near-duplicate image flagging. |
| `log_level` | `INFO` | `AYNVQA_LOG_LEVEL` | Logging verbosity. |

### ASR / M2

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `asr_sample_n` | `50` | `AYNVQA_ASR_SAMPLE_N` | Sample size for the M2 ASR bench. |
| `whisper_model_size` | `medium` | `AYNVQA_WHISPER_MODEL_SIZE` | Local Whisper model size — chosen for more consistent punctuation and a higher "options are" marker-match rate than `small`, at ~1.9x the latency. |
| `whisper_device` | `cpu` | `AYNVQA_WHISPER_DEVICE` | Device for the primary Whisper pass. |
| `whisper_compute_type` | `int8` | `AYNVQA_WHISPER_COMPUTE_TYPE` | Quantization for the primary Whisper pass. |
| `fanar_api_key` | `None` | `AYNVQA_FANAR_API_KEY` | Optional — Fanar Aura-STT backend, only used if set. |
| `fanar_base_url` | `https://api.fanar.qa/v1` | `AYNVQA_FANAR_BASE_URL` | Fanar API base URL. |
| `fanar_stt_model` | `Fanar-Aura-STT-1` | `AYNVQA_FANAR_STT_MODEL` | Fanar model name. |
| `openai_api_key` | `None` | `AYNVQA_OPENAI_API_KEY` | Optional — OpenAI transcription backend, only used if set. |
| `openai_base_url` | `https://api.openai.com/v1` | `AYNVQA_OPENAI_BASE_URL` | OpenAI API base URL. |
| `openai_transcribe_model` | `gpt-4o-transcribe` | `AYNVQA_OPENAI_TRANSCRIBE_MODEL` | OpenAI transcription model name. |

### M3: local VLM / Ollama

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `pipeline_sample_n` | `None` (whole split) | `AYNVQA_PIPELINE_SAMPLE_N` | Items to run the full cascade over. |
| `ollama_base_url` | `http://localhost:11434` | `AYNVQA_OLLAMA_BASE_URL` | Local Ollama server URL — no API key needed. |
| `ollama_parse_model` | `qwen2.5vl:7b` | `AYNVQA_OLLAMA_PARSE_MODEL` | Transcript-parsing model. |
| `ollama_select_model` | `qwen3-vl:8b` | `AYNVQA_OLLAMA_SELECT_MODEL` | Joint-MCQ selection model (M6 default). |

### M4: pipeline-robustness repair escalation

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `repair_enabled` | `True` | `AYNVQA_REPAIR_ENABLED` | Master gate for the 3-step repair ladder. |
| `repair_whisper_model_size` | `large-v3` | `AYNVQA_REPAIR_WHISPER_MODEL_SIZE` | Stronger ASR model used at step 2 of repair. |
| `repair_whisper_device` | `cuda` | `AYNVQA_REPAIR_WHISPER_DEVICE` | Device for the repair-stage Whisper model. |
| `repair_whisper_compute_type` | `float16` | `AYNVQA_REPAIR_WHISPER_COMPUTE_TYPE` | Compute precision — `float16`, not `int8`, since this runs on GPU where float16 is both faster and more accurate. |

### M5: chain-of-thought select prompt

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `cot_enabled` | `False` | `AYNVQA_COT_ENABLED` | Chain-of-thought select-prompt switch. Off — a 43-item pilot looked promising (30.2% recovery) but the full-scale ablation netted only +0.2 points; not validated as a net win. |

### M5: few-shot exemplar retrieval

| Field | Default | Env var | Purpose |
|---|---|---|---|
| `fewshot_enabled` | `False` | `AYNVQA_FEWSHOT_ENABLED` | Few-shot exemplar retrieval from `train`. Off — same reason as `cot_enabled`; full-scale ablation was an exact wash (+0.0 points). |
| `fewshot_k` | `2` | `AYNVQA_FEWSHOT_K` | Exemplars retrieved per query when enabled. |
| `fewshot_num_ctx` | `16384` | `AYNVQA_FEWSHOT_NUM_CTX` | Raises Ollama's context window for few-shot calls only — Ollama's own 4096 default times out (400s) on more than one image in a prompt. |

### Cross-field notes

- `ollama_parse_model` and `ollama_select_model` started identical (both `qwen2.5vl:7b`) but diverged in M6 — only the select model was bumped after an ablation showed a meaningful gain on that specific stage.
- `repair_whisper_*` only takes effect when `repair_enabled=True`; it deliberately configures a stronger/different setup (`large-v3`/`cuda`/`float16`) than the base ASR pass (`medium`/`cpu`/`int8`).
- `fewshot_k` and `fewshot_num_ctx` are only meaningful when `fewshot_enabled=True`.
- `cot_enabled` and `fewshot_enabled` share the same rationale for defaulting `False`: both are M5 features that showed pilot-scale promise but did not hold up at full-dataset scale, so both remain opt-in for ablation runs rather than on by default.

---

## 7. CLI Guide

Four CLI entry points exist, all defined in `pyproject.toml`'s `[project.scripts]`. No `scripts/` directory and no other CLI or standalone script exists anywhere in `src/`.

```toml
[project.scripts]
aynvqa-audit      = "ayn_vqa.audit.run_audit:main"
aynvqa-predict    = "ayn_vqa.run_baseline:main"
aynvqa-transcribe = "ayn_vqa.run_asr_bench:main"
aynvqa-run        = "ayn_vqa.run_pipeline:main"
```

Each is equivalently invocable as `python -m ayn_vqa.<module>` (e.g. `python -m ayn_vqa.run_pipeline`), though `uv run aynvqa-X` is the documented form throughout this project.

**Shared conventions across all four CLIs:**
- `--split` choices: `train`, `dev`, `devtest` (default `dev` on the three that have it). `devtest` has no gold labels, so scoring is skipped for it.
- `--language` choices: `en`, `msa` (default `msa` everywhere — the competition track).
- `--seed` overrides `AYNVQA_RANDOM_SEED` — present on `aynvqa-predict`, `aynvqa-transcribe`, `aynvqa-run`, but **absent on `aynvqa-audit`** (its sample-grid seed is env-only, no CLI flag).
- `--data-root` overrides `AYNVQA_DATA_ROOT` on every CLI that has it.
- `--log-level` overrides `AYNVQA_LOG_LEVEL` on every CLI that has it.

---

### 7.1 `aynvqa-audit` — M0 data audit

**Purpose:** Orchestrates every M0 audit module in one pass per split: load JSONL → validate media paths exist → probe audio/image stats → hash images for exact/near duplicates → render a random sample contact sheet → write CSV/JSON/Markdown reports. Logs to both console and `<output-dir>/audit.log`.

| Flag | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--language` | str | `msa` | `en`, `msa` | Audio track to audit. |
| `--splits` | str, nargs+ | `train dev devtest` | `train`, `dev`, `devtest` | Splits to audit. |
| `--data-root` | Path | `None` | — | Override `AYNVQA_DATA_ROOT`. |
| `--output-dir` | Path | `None` | — | Override the report output directory (default: `reports/m0_data_audit`). |
| `--sample-n` | int | `None` | — | Override the sample grid size (`AYNVQA_SAMPLE_GRID_N`). |
| `--near-dup-max-distance` | int | `None` | — | Override the near-duplicate Hamming threshold. |
| `--log-level` | str | `None` | — | Override the log level. |

Note: there is no `--seed` flag on this CLI — its sample-grid seed is always `AYNVQA_RANDOM_SEED`, settable only via env/`.env`.

**Typical usage:** run with no arguments for the full `train`+`dev`+`devtest` audit.

**Example:**
```bash
uv run aynvqa-audit
```

---

### 7.2 `aynvqa-predict` — M1 trivial-baseline harness

**Purpose:** Runs a trivial baseline selector (`random` or `constant`) over one split: load → predict → write `prediction_<name>_<split>_<language>.csv` → format-check → (if labeled) score and log to `experiments.md`. Exists to prove the load→predict→submit→score loop works before any real model is involved.

| Flag | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--selector` | str | `random` | `random`, `constant` | Which trivial baseline to run. |
| `--constant-value` | int | `0` | `0`, `1`, `2` | Only used with `--selector constant`. |
| `--split` | str | `dev` | `train`, `dev`, `devtest` | `train`/`dev` are scored against gold labels; `devtest` is not. |
| `--language` | str | `msa` | `en`, `msa` | — |
| `--data-root` | Path | `None` | — | Override `AYNVQA_DATA_ROOT`. |
| `--output-dir` | Path | `None` | — | Override the prediction output directory (default: `reports/m1_baselines`). |
| `--seed` | int | `None` | — | Override `AYNVQA_RANDOM_SEED`; only used by `--selector random`. |
| `--log-level` | str | `None` | — | — |

`experiments.md` is hardcoded to `PROJECT_ROOT/experiments.md` — there is no flag to redirect it.

**Examples:**
```bash
uv run aynvqa-predict --selector random --split dev
uv run aynvqa-predict --selector constant --constant-value 0 --split dev
```

---

### 7.3 `aynvqa-transcribe` — M2 ASR bench

**Purpose:** Runs one ASR backend over a seeded sample of one split, caching every transcript to `artifacts/<split>/asr/<language>_<config_key>.jsonl`, then prints a console summary: success/failure counts, latency stats (mean/min/max), transcript length stats, and up to 5 full sample transcripts for by-ear spot-checking. Writes no report directory — its only artifact is the ASR JSONL cache.

| Flag | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--backend` | str | `whisper` | `whisper`, `fanar`, `openai` | ASR backend to use. |
| `--split` | str | `dev` | `train`, `dev`, `devtest` | — |
| `--language` | str | `msa` | `en`, `msa` | — |
| `--sample-n` | int | `None` | — | Override `AYNVQA_ASR_SAMPLE_N` (default `50`). |
| `--seed` | int | `None` | — | Override `AYNVQA_RANDOM_SEED`. |
| `--data-root` | Path | `None` | — | — |
| `--artifacts-dir` | Path | `None` | — | — |
| `--whisper-model-size` | str | `None` | — | Override `AYNVQA_WHISPER_MODEL_SIZE` (only used with `--backend whisper`). |
| `--log-level` | str | `None` | — | — |

**Quirk to be aware of:** `sample_n = args.sample_n or settings.asr_sample_n` uses Python truthiness, not an `is not None` check — so `--sample-n 0` silently falls back to the configured default (`50`) instead of sampling zero records. (`aynvqa-run`'s `--sample-n` does not have this quirk — it uses `is not None`.)

**Backends not fully wired to CLI flags:** `--backend fanar` and `--backend openai` read their API keys, model names, and base URLs entirely from `Settings`/env — there is no `--fanar-*` or `--openai-*` CLI override for any of it. Per the project's own characterization, these backends are "ready-but-unexercised."

**Examples:**
```bash
uv run aynvqa-transcribe --backend whisper --whisper-model-size medium --split dev
uv run aynvqa-transcribe --whisper-model-size large-v3
```

---

### 7.4 `aynvqa-run` — the full cascade (largest / most important CLI)

**Purpose:** The real end-to-end pipeline: load a split → transcribe (Whisper) → parse (Ollama VLM) → optionally repair → optionally retrieve few-shot exemplars → select (Ollama joint-MCQ VLM) → write submission CSV → format-check → (if labeled) score + log → build a per-record error-analysis CSV. Every stage's output is cached, so re-running after a crash or config tweak recomputes only what changed. Producing `prediction.csv` is as far as this goes — Codabench upload remains manual.

| Flag | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--split` | str | `dev` | `train`, `dev`, `devtest` | — |
| `--language` | str | `msa` | `en`, `msa` | — |
| `--sample-n` | int | `None` (whole split) | — | Run only a seeded sample of this size. |
| `--seed` | int | `None` | — | Override `AYNVQA_RANDOM_SEED`. |
| `--data-root` | Path | `None` | — | — |
| `--artifacts-dir` | Path | `None` | — | — |
| `--output-dir` | Path | `None` | — | Default: `reports/m3_pipeline`. |
| `--whisper-model-size` | str | `None` | — | Override `AYNVQA_WHISPER_MODEL_SIZE`. |
| `--ollama-parse-model` | str | `None` | — | Override `AYNVQA_OLLAMA_PARSE_MODEL`. |
| `--ollama-select-model` | str | `None` | — | Override `AYNVQA_OLLAMA_SELECT_MODEL`. |
| `--ollama-base-url` | str | `None` | — | Override `AYNVQA_OLLAMA_BASE_URL`. |
| `--repair-enabled` / `--no-repair-enabled` | bool flag | `None` | — | Override `AYNVQA_REPAIR_ENABLED` — use `--no-repair-enabled` for an ablation run. |
| `--cot-enabled` / `--no-cot-enabled` | bool flag | `None` | — | Override `AYNVQA_COT_ENABLED` — use `--cot-enabled` for an ablation run. |
| `--fewshot-enabled` / `--no-fewshot-enabled` | bool flag | `None` | — | Override `AYNVQA_FEWSHOT_ENABLED` — use `--fewshot-enabled` for an ablation run. |
| `--fewshot-k` | int | `None` | — | Override `AYNVQA_FEWSHOT_K`. |
| `--log-level` | str | `None` | — | — |

**Settings with no corresponding CLI flag on `aynvqa-run` at all** (env/`.env` only): `repair_whisper_model_size`, `repair_whisper_device`, `repair_whisper_compute_type`, `fewshot_num_ctx`, `fanar_*`/`openai_*` (this pipeline's ASR backend is always `WhisperLocalASR`; `aynvqa-run` never builds a Fanar or OpenAI backend).

`experiments.md` is hardcoded — no redirect flag. The process exits with `SystemExit(1)` if the format checker fails.

**Typical usage/discipline:** run a small `--sample-n` first to sanity-check a config change cheaply, then the full split. Ablation runs toggle exactly **one** flag at a time (e.g. only `--no-repair-enabled`, or only `--ollama-select-model`), holding everything else fixed, so the effect of that one variable can be cleanly isolated — this discipline is followed explicitly in every M4/M5/M6 comparison in this project.

**Examples:**
```bash
# Small sample first, then full run
uv run aynvqa-run --split dev --sample-n 50
uv run aynvqa-run --split dev

# M4 ablation: repair off vs on
uv run aynvqa-run --split dev --language msa --no-repair-enabled
uv run aynvqa-run --split dev --language msa --repair-enabled

# M6: select-model swap (repair stays on via its default, not an explicit flag)
uv run aynvqa-run --split dev --language msa --ollama-select-model qwen3-vl:8b

# A fully-explicit example touching every M4/M5/M6-relevant flag at once
uv run aynvqa-run \
  --split dev --language msa \
  --ollama-select-model qwen3-vl:8b \
  --repair-enabled \
  --cot-enabled \
  --fewshot-enabled --fewshot-k 2 \
  --output-dir reports/m6_vlm_swap \
  --sample-n 50
```

---

## 8. Running the full pipeline

A complete, from-scratch walkthrough, step by step, using the actual production defaults (repair on, parser v2, `qwen3-vl:8b` select model):

**1. Install and configure** (see §4, §6):
```bash
uv sync --group dev --group notebook
cp .env.example .env   # edit AYNVQA_DATA_ROOT if needed
```

**2. Audit the raw dataset** (optional, but recommended before anything else — this is how the project itself caught 3 unmaterialized Git-LFS image stubs early):
```bash
uv run aynvqa-audit
```
Inspect `reports/m0_data_audit/` for the generated CSV/JSON/Markdown reports and contact-sheet figures.

**3. Establish the trivial baselines** (sanity-checks the load→predict→submit→score loop, and gives you a chance-floor / label-skew reference point):
```bash
uv run aynvqa-predict --selector random --split dev
uv run aynvqa-predict --selector constant --constant-value 0 --split dev
```

**4. Bench the ASR backend** (optional — the project already settled on `whisper-medium` after M2, but this is how to reproduce or extend that comparison):
```bash
uv run aynvqa-transcribe --backend whisper --whisper-model-size medium --split dev
```

**5. Pull the required Ollama models and start the server** (see §5):
```bash
ollama pull qwen2.5vl:7b
ollama pull qwen3-vl:8b
ollama serve
```

**6. Run the full cascade — small sample first:**
```bash
uv run aynvqa-run --split dev --sample-n 50
```
Inspect the console output and `reports/m3_pipeline/` for `prediction_*.csv` and `error_analysis_*.csv`. Because `repair_enabled=True` and `ollama_select_model=qwen3-vl:8b` are already the config defaults, this 50-item run already reflects the current production configuration — no extra flags needed.

**7. Run the full 500-item `dev` split:**
```bash
uv run aynvqa-run --split dev
```
Expect roughly two hours of wall-clock time on a fresh cache (the primary Whisper `medium` CPU pass and the repair-escalation ladder dominate runtime — see §10 for exact per-milestone runtimes). A cached re-run of an identical configuration completes in seconds.

**8. Review results:**
- `experiments.md` gets a new row with `accuracy`/`balanced_accuracy`/`macro_f1`.
- `reports/m3_pipeline/error_analysis_*.csv` has one row per record with transcript, parse, prediction, gold, and `option_quality_ok`/`option_quality_reasons` columns.
- Console output prints per-country and per-category accuracy breakdowns, plus the pipeline-artifact-vs-genuine-miss split via `summarize_option_quality`.

**9. Run an ablation** if evaluating a change (see §7.4, §12, §16 for the discipline this project follows — one variable at a time, small sample first, full split before adopting):
```bash
uv run aynvqa-run --split dev --sample-n 50 --no-repair-enabled   # quick check
uv run aynvqa-run --split dev --no-repair-enabled                  # full-scale confirmation
```

**10. Manual step: submit to Codabench.** This pipeline stops at `prediction.csv`. Uploading it to the competition's Codabench page is not automated by this repository.

---

## 9. Caching system

### Layout

```
artifacts/<split>/<stage>/<language>_<config_key>.jsonl
```
`<stage>` ∈ `{asr, parse, select}`. Retrieval's ASR/parse hydration (for few-shot) reuses the `asr`/`parse` stage functions unmodified, just pointed at `train`'s own namespace.

### Primitives (`artifacts.py`)

```python
artifact_path(artifacts_dir, split, stage, config_key) -> artifacts_dir/split/stage/f"{config_key}.jsonl"
read_jsonl_cache(path, id_field="record_id") -> {row[id_field]: row}   # missing file -> {}, never an error
append_jsonl(path, row)                                                 # one line, flushed immediately
```
`append_jsonl` writes and flushes **one record at a time** — a long ASR/VLM run killed partway through keeps every result computed before the interruption; the next run only recomputes what's missing.

### How cache keys are built, per stage

- **ASR:** `f"whisper-{whisper_model_size}"`, e.g. `whisper-medium`. Cache file: `<language>_whisper-medium.jsonl`.
- **Parse:** `parser.name`, which for `OllamaTranscriptParser` is the **hardcoded, static string `"ollama-parse"`** — it does **not** vary with the underlying model or the prompt template text. This is a deliberate load-bearing caveat, explained below.
- **Select:** starts as `selector.name` (which for `OllamaJointMCQSelector` *does* vary with model and CoT), then `run_pipeline` conditionally appends `__repair` and/or `__fewshot{k}` suffixes.
- The submission/error-analysis file names combine all three: `f"{asr_config_key}__{parser_config_key}__{effective_selector_config_key}"`.

### Why cache isolation exists

This project hit the same class of bug — *two runs that actually produce different output silently collide on one cache file and replay each other's stale results* — more than once. Each isolation mechanism below exists because of a specific real incident or a near-miss caught before it happened, not defensive design in the abstract.

**Repair isolation (`__repair` suffix).** A repair-off and repair-on run would otherwise collide on the same select-stage cache path, even though repair changes the parsed question/options fed into select for flagged records. Without the suffix, re-running either configuration a second time would silently read the *other* configuration's cached predictions. Repair's own two internal escalation steps get their own isolated parse-cache keys too (`{parser_config_key}-repair1`, `{parser_config_key}-repair2`), and the stronger repair-tier ASR call uses a wholly separate config key (e.g. `whisper-large-v3`) so it never collides with primary-tier `whisper-medium` transcripts.

**CoT isolation (baked into `selector.name`, no extra suffix needed).** `OllamaJointMCQSelector.__init__` sets `self.name = f"{base_name}-cot"` when CoT is on. Because CoT is a static property of the selector instance — identical for every record in a run — it can live in the object's own `.name` rather than needing a per-run suffix computed by the orchestrator, and a CoT-on/CoT-off run automatically land on different files.

**Few-shot isolation (`__fewshot{k}` suffix, applied after `__repair`).** Few-shot changes what selection sees on a *per-record* basis (exemplars come from a separate retrieval stage, not a static property of the selector), so it can't just live in `.name` the way CoT does — it needs its own orchestrator-level suffix. This also folds in `fewshot_k` itself, so a `k=2` run and a `k=3` run get distinct cache files.

**Select-stage model-name isolation.** `OllamaJointMCQSelector.name` includes a model-slug suffix whenever the configured model differs from the hardcoded default (`qwen2.5vl:7b`) — e.g. `ollama-joint-mcq-qwen3-vl-8b`. **This is the M6 lesson, made concrete:** before this fix, the selector's cache key varied with the CoT flag but never with the model itself. A bare `--ollama-select-model qwen3-vl:8b` swap would have silently collided with the default model's existing select-stage cache — replaying `qwen2.5vl:7b`'s old predictions and never actually invoking `qwen3-vl:8b` at all. Format-checking and scoring would have looked completely normal, silently showing "no change" instead of erroring. This was **caught and fixed before running**, not discovered after a wasted 2-hour run, and is covered by two new tests. Because the current production default is already `qwen3-vl:8b` (not the hardcoded `_DEFAULT_MODEL = "qwen2.5vl:7b"`), the selector's cache key already includes the model slug in a fresh checkout today — this is not a dormant risk you'd need to remember to trigger.

### The parser-v2 cache lesson (the one gap that still exists)

**Unlike every mechanism above, the parse stage has no automatic isolation for prompt-text or model changes.** `OllamaTranscriptParser.name = "ollama-parse"` is a class-level constant — it never varies with the model used for parsing, nor with the content of the prompt template string. When the parse prompt was revised from v1 to v2 in M4 (v1 told the model to fabricate a plausible third option when one couldn't be recovered — hand review found this caused real hallucination; v2 instead targets specific structural failure modes without fabricating), the cache key (`parser.name`) stayed exactly `"ollama-parse"` — identical to before the edit. Since the cache is keyed purely by this string, the pre-existing v1 cache file was **still valid by path** and would have been silently replayed against v2, serving stale v1-parsed rows (including v1's fabricated-option failure mode) under what looked like a v2 run.

**The mitigation actually used was not a code fix** — it was a manual, one-time cache archival step: moving the old `ollama-parse.jsonl` cache files aside (into `artifacts/_v1_prompt_archive/`, still present in this repo today) before running the v2-instrumented pipeline, so v2 calls would execute fresh instead of hitting the stale v1 cache. This precedent is exactly what motivated building *automatic* code-level isolation into repair, CoT, and few-shot the moment they were built. **The parse stage itself still lacks this protection today.**

### How to safely clear cache

- To force a full re-run of a specific stage/config, delete (or move aside, as was done for the v1→v2 archival) the specific `artifacts/<split>/<stage>/<language>_<config_key>.jsonl` file — never delete the whole `artifacts/` tree unless you intend to lose every cached run.
- Prefer archiving (rename/move) over deleting when the old cache might still be useful for comparison, as the `_v1_prompt_archive/` directory demonstrates.

### How to avoid stale-cache bugs (a real lesson, not generic advice)

**If you ever edit `stages/parse.py`'s prompt text, or change `settings.ollama_parse_model` to a different model, you must manually archive or delete the existing `artifacts/<split>/parse/*_ollama-parse.jsonl` cache files before re-running** — the code will not detect the change and will not warn you. A parse-stage cache hit is only trustworthy if you know the prompt and model haven't changed since it was written. This is the single sharpest caching caveat in the whole project; every other stage (ASR model size, select model, CoT, repair, few-shot-k) is already protected against this class of bug automatically.

---

## 10. Milestones

### M0 — Data audit

- **Goal:** Understand the raw Task 1a data (schema, file integrity, duplicates) before any ASR/OCR/VLM code is written.
- **What was implemented:** `config.py`, `data/schema.py` (`Task1aRecord`, frozen, `extra="forbid"`), `data/loader.py` (tolerant line-by-line JSONL parser), `data/validation.py`, `audit/audio_stats.py`, `audit/image_stats.py`, `audit/hashing.py` (SHA-256 exact-dup + dHash near-dup), `audit/sampling.py`, `audit/report.py`, `audit/run_audit.py` (`aynvqa-audit` CLI), `notebooks/00_data_audit.ipynb`, 30 tests.
- **Key files:** `src/ayn_vqa/audit/`, `src/ayn_vqa/data/`.
- **Results:** 3 of 4,000 MSA-track images were unmaterialized Git-LFS pointer stubs (134-byte text files), not real images — fixed via `git lfs pull --include "images/*"`. 0 exact duplicates; 3 near-duplicate image pairs across the full MSA track.
- **Lessons learned:** never sample via `records[:n]` — train/dev JSONL is sorted alphabetically by country, so a slice would sample one country only; use `random.Random(seed).sample` instead.
- **Design decisions:** separate git repo (not inside the dataset clone) referencing data by path only, never copying it; `extra="forbid"` schema so a surprise field in the future blind `test` split fails loudly; pandas nullable `"Int64"` dtype for `label` to avoid `None`→`float64` print artifacts.
- **Status:** Complete/foundational.

### M1 — Repo skeleton + trivial baselines

- **Goal:** Prove the whole harness (load → predict → write submission → format-check → score → log) works end to end before any real model exists.
- **What was implemented:** vendored `scorer_official/`, `stages/select.py` (`RandomSelector`, `ConstantSelector`), `submission.py`, `evaluate.py`, `experiments.py`, `run_baseline.py` (`aynvqa-predict` CLI).
- **Key files:** `src/ayn_vqa/stages/select.py`, `src/ayn_vqa/run_baseline.py`, `src/ayn_vqa/evaluate.py`.
- **Results (dev, msa, 500 items):** `random`: accuracy 0.3340, balanced_accuracy 0.3322, macro_f1 0.3320. `constant-0`: accuracy 0.3580, balanced_accuracy 0.3333, macro_f1 0.1757 (label distribution 35.8%/31.6%/32.6%).
- **Lessons learned:** `constant-0`'s balanced_accuracy is exactly 1/3 (perfect class-0 recall, zero elsewhere) while macro-F1 is much lower (0.1757, precision-driven) — a concrete illustration of why accuracy alone is misleading, referenced again in M4. A first draft of `test_select.py` built a fresh `RandomSelector` per record inside a list comprehension — every "run" replayed the RNG's first draw 50 times, not a real sequence — fixed by constructing one selector instance per run.
- **Status:** Complete/foundational.

### M2 — ASR bench

- **Goal:** Pick a transcription stack with data, not vibes.
- **What was implemented:** `stages/asr.py` (`WhisperLocalASR`, `FanarAuraASR`, `OpenAITranscribeASR`), `artifacts.py` (generic JSONL cache), `run_asr_bench.py` (`aynvqa-transcribe` CLI), `asr_compare.py`.
- **Key files:** `src/ayn_vqa/stages/asr.py`, `src/ayn_vqa/artifacts.py`.
- **Results (dev, msa, seeded 50-item sample):** whisper-medium: 50/50 ok, mean 114.4 chars, mean latency 7.92s, "؟" present 28%, "الخيارات هي" marker present 82%. whisper-small: 50/50 ok, mean 114.2 chars, mean latency 4.23s, "؟" present 22%, marker present 78%.
- **Lessons learned:** content-word errors exist on both sizes (neither is uniformly better) — confirms the need for native-speaker judgment, not just aggregate metrics. Windows/Rich console `UnicodeEncodeError` on Arabic (cp1252 codepage) fixed via `legacy_windows=False` + forced UTF-8 stdout/stderr.
- **Design decision:** default to `whisper-medium` for M3 (better punctuation/marker consistency), keep `whisper-small` available for fast iteration.
- **Status:** Recommendation adopted into M3.

### M3 — First real pipeline

- **Goal:** First end-to-end cascade that actually looks at image + audio to answer the question.
- **What was implemented:** local Qwen2.5-VL-7B via Ollama (found an idle NVIDIA RTX 2000 Ada, 16GB VRAM); `ollama_client.py`, `stages/parse.py` (`OllamaTranscriptParser`), `stages/select_vlm.py` (`OllamaJointMCQSelector`, joint-MCQ, explicitly designed to avoid the official baseline's brittle `re.search(r"[012]", ...)` regex parser), `error_analysis.py`, `run_pipeline.py` (`aynvqa-run` CLI).
- **Key files:** `src/ayn_vqa/stages/parse.py`, `src/ayn_vqa/stages/select_vlm.py`, `src/ayn_vqa/run_pipeline.py`.
- **Results (dev, msa, full 500):** accuracy **71.60% (358/500)**, balanced_accuracy 0.7141, macro_f1 0.7143. vs. official baseline 39.8% (MSA) / 66.4% (English): +32 points over the MSA baseline, +5 points over the baseline's own English number. 500/500 ASR succeeded, 500/500 parsed, 500/500 predicted (0 fallbacks). Runtime ~2h4m. Per-country: Palestine weakest (58.6%, n=29), Sudan strongest (93.1%, n=29). Per-category: weakest "Objects, Materials & Clothing" (58.7%) and "Food & Cooking" (59.4%); strongest "History, Geography & National Identity" (83.6%).
- **Lessons learned:** a prior seeded 50-item sample measured 82.00% — inside the statistical confidence interval for that sample size, but optimistic vs. the full-split number; 50-item samples are for *directional* comparisons only, full-split runs for the reportable number.
- **Status:** Adopted as production pipeline / new baseline for M4 comparison.

### M4 — Pipeline robustness

- **Goal:** M3's "142 wrong = 142 genuine VLM misses" conclusion was wrong — a hand-audit found the parser's fixed 3-option schema fabricates content when a transcript doesn't cleanly contain 3 options. 69 of 142 (48.6%) were silent pipeline corruption, not visual-reasoning failures.
- **What was implemented:** `option_quality.py` (`check_option_quality()`, `check_repetition_loop()`), `error_analysis.py` extended with option-quality summaries, a 3-step repair escalation ladder in `run_pipeline.py`, `stages/parse.py`'s `_STRICT_PROMPT`, and the Windows CUDA DLL fix in `stages/asr.py`. 106 tests total.
- **Key files:** `src/ayn_vqa/option_quality.py`, `src/ayn_vqa/run_pipeline.py`, `src/ayn_vqa/stages/asr.py`.
- **Results:** repair off (=M3): 71.60% (358/500). Repair on, v1 prompt: **73.80% (369/500)**, net +11 correct. Repair on, **prompt v2**: **74.80% (374/500)**, net +5 vs v1. Repair funnel (v2): 500 records → 125 flagged degenerate → 33 resolved by reparse / 42 by ASR escalation / 50 still degenerate at run's end. Whisper-medium (CPU) vs large-v3 (GPU): 7.92s vs **1.32s** mean latency — large-v3 is ~6x faster once on GPU, not slower.
- **Lessons learned:** (1) structural validity ≠ semantic correctness — a confidently-invented plausible option passes the quality check; (2) the automated `option_quality` checker and careful manual review disagree ~30% of the time (42/142 on the original error set); (3) repair-retry can reorder options relative to the spoken sequence, risking gold-label/option misalignment — found by hand, not caught by any test; (4) VLM selection has ~0.7% resampling noise even at temperature=0.0; (5) the stricter reparse prompt is inconsistent and can make an already-acceptable parse worse on retry.
- **Design decision, deferred:** the fixed three-non-empty-option schema is a known limitation (some transcripts honestly support only 2 options) — explicitly not changed, to preserve comparability with M0–M4 results.
- **Status: FROZEN.** Production defaults: `repair_enabled=True`, parser prompt v2, Whisper large-v3 as repair-escalation backend. **Frozen baseline: 74.80% (374/500), dev/msa.**

### M5 — CoT and few-shot retrieval

- **Goal:** Address the frozen baseline's genuine visual-reasoning misses (43/126 wrong = 34.1% by reconciled taxonomy). Ceiling if fully solved: +8.6 points.
- **What was implemented:** `stages/select_vlm.py`'s `use_cot` flag (adds `visible_details`+`reasoning` fields, distinct cache name `ollama-joint-mcq-cot`); `stages/retrieve.py` (new — k=2 exemplar retrieval via VLM category/subcategory classification over a 9-category/31-subcategory taxonomy); `OllamaClient.chat_messages` for multi-turn few-shot prompting; `fewshot_num_ctx` raised to 16384.
- **Key files:** `src/ayn_vqa/stages/select_vlm.py`, `src/ayn_vqa/stages/retrieve.py`.
- **Results:** CoT: **75.00% (375/500)**, +1 net correct (32 fixed / 31 regressed) — vs. a 43-item pilot that recovered 13/43 (30.2%). Few-shot (k=2): **74.80% (374/500)**, +0 net (18 fixed / 18 regressed, exact wash) — vs. a pilot that recovered 6/43 (14.0%).
- **Lessons learned:** (1) a pilot on the hard subset doesn't reliably predict full-scale net effect when the technique changes behavior on *every* call, not a targeted subset; (2) reading reasoning traces (not just counting flips) found a genuine reasoning-to-answer-index inconsistency bug (≥7 clear cases where the `reasoning` field names the correct option but `answer_index` doesn't match) that pure accuracy numbers would have hidden; (3) a "fabricated-placeholder seduction" pattern (≥4 cases) where CoT carefully reasons about, and selects, a known-degenerate placeholder option; (4) small-sample retrieval-quality diagnosis doesn't automatically generalize — the pilot's "good match → success" correlation didn't hold at 500 items.
- **Status: Neither adopted.** Both `cot_enabled` and `fewshot_enabled` remain `False` by default. **M4's 74.80% (374/500) remained the reference baseline going into M6.**

### M6 — VLM swap

- **Goal:** M5 found prompt-level techniques hit diminishing returns; test a different lever — the underlying VLM itself, pipeline otherwise unchanged.
- **What was implemented:** model-selection research (InternVL checked directly against Ollama's official catalog and found **not available** — only unofficial community GGUF conversions exist, ruled out); Qwen3-VL:8b selected (first-party official Ollama model, 6.14GB); a 5-item smoke test passed cleanly before the full run; **a real cache-collision bug fixed before running** — `OllamaJointMCQSelector.name` varied with the CoT flag but never with `model`, which would have silently replayed `qwen2.5vl:7b` predictions under a `qwen3-vl:8b` label. Fixed to include the model name in `.name`, covered by two new tests.
- **Key files:** `src/ayn_vqa/stages/select_vlm.py`, `src/ayn_vqa/config.py`.
- **Results (dev, msa, full 500, repair enabled, parser v2, no CoT, no few-shot, only `--ollama-select-model qwen3-vl:8b`):** **79.00% (395/500)** — **+4.2 points, +21 net correct** over the frozen M4 baseline, the largest single-change improvement of M4–M6. 48 flipped wrong→correct, 27 flipped correct→wrong. Runtime ~2h14m, the longest full run of the project so far.
- **Lessons learned:** ≥10 of the 27 regressions also regressed under CoT and/or few-shot in M5 (several to the identical wrong index) — one item regressed under all three techniques independently, suggesting a genuinely hard core of items rather than technique-specific noise. `nvidia-smi` corrected a hardware-naming discrepancy: the GPU is actually an RTX 2000 Ada Generation, not the RTX 4080 that had been referenced when the M4 repair backend was configured — same VRAM capacity, but a lower-throughput card, relevant context for qwen3-vl's markedly higher/slower latency.
- **Decision:** per the project's adoption criterion (a single model giving clear improvement gets adopted), `ollama_select_model` in `config.py` now defaults to `qwen3-vl:8b`. `ollama_parse_model` unchanged (parser-model swap unevaluated).
- **Status: ADOPTED.** New frozen baseline for all future milestones: **79.00% (395/500), dev/msa.**

### Milestone comparison table

| Milestone | Configuration | Accuracy | Status |
|---|---|---:|---|
| M1 | `random` baseline | 33.40% (167/500) | Reference (chance floor) |
| M1 | `constant-0` baseline | 35.80% (179/500) | Reference (label skew) |
| M3 | Whisper-medium → parse → joint-MCQ (Qwen2.5-VL-7B), no repair | 71.60% (358/500) | Adopted → superseded by M4 |
| M4 | + repair ladder, prompt v1 | 73.80% (369/500) | Superseded by v2 |
| M4 | + repair ladder, prompt **v2** | **74.80% (374/500)** | **FROZEN** (was baseline through M5) |
| M5 | + chain-of-thought | 75.00% (375/500) | Evaluated, **not adopted** |
| M5 | + few-shot (k=2) | 74.80% (374/500) | Evaluated, **not adopted** |
| M6 | select model → `qwen3-vl:8b` | **79.00% (395/500)** | **ADOPTED — current frozen baseline** |

---

## 11. Current production pipeline

As of M6, the frozen, default production configuration is:

| Setting | Value | Why |
|---|---|---|
| `repair_enabled` | `True` | M4 measured +1.0 to +2.2 point gains from repair (depending on prompt version) with a well-understood, well-isolated escalation ladder; the cost (extra runtime, only on flagged records) was judged worth the accuracy gain and the reduction in silent pipeline-artifact errors. |
| `ollama_parse_model` | `qwen2.5vl:7b` (**prompt v2**) | Never swapped — only the select stage was evaluated for a model change in M6. The prompt itself was revised from v1 to v2 in M4 after v1's tendency to fabricate a plausible-but-invented third option was found via manual review; v2 targets specific structural failure modes (comma-boundary recovery, boilerplate-prefix stripping, order preservation) without fabricating, and measured +1.0 point / 5 fewer wrong on the repair path. |
| `whisper_model_size` (primary) | `medium` | M2's bench found more consistent punctuation and a higher "options are" marker-match rate than `small`, at acceptable extra latency (CPU). |
| `repair_whisper_model_size` (escalation) | `large-v3` | Used only for the small subset of records that reach repair step 2; runs on GPU where it measured ~6x *faster* than the CPU `medium` pass (1.32s vs 7.92s mean latency), not slower — so escalating to a stronger model here is essentially free once GPU is available. |
| `ollama_select_model` | `qwen3-vl:8b` | M6's full 500-item ablation measured +4.2 points (79.00% vs 74.80%) — the single largest accuracy gain of the entire project, and the largest lever available after M5 found prompt-level techniques (CoT, few-shot) had hit diminishing returns. |
| `cot_enabled` | `False` | M5's full-scale ablation netted only +0.2 points (75.00% vs 74.80%) despite a promising 30.2%-recovery pilot — the discrepancy traced to a genuine reasoning-to-answer-index reliability bug, not a real capability gain. Not worth the added latency and complexity as currently implemented. |
| `fewshot_enabled` | `False` | M5's full-scale ablation was an exact wash (+0.0 points, 18 fixed / 18 regressed) — traced to category-classification precision being the real bottleneck, not the retrieval mechanism itself. |

This is the exact configuration `uv run aynvqa-run --split dev` runs today with zero extra flags — every value above is `config.py`'s baked-in default.

---

## 12. Evaluation methodology

### The benchmark

Every headline number in this project is measured on the **full 500-item `dev`/`msa` split**, scored by the vendored, unmodified official scorer (`scorer_official/scorer/score.py`) — the same code that would score a real Codabench submission. `train`/`dev` carry gold labels; `devtest` (and the not-yet-released blind `test`) do not, so scoring is automatically skipped for them.

### Manual taxonomy work

Automated metrics alone were repeatedly found insufficient in this project's own history. M4's `option_quality.py` heuristic and careful manual review of the same error set disagreed **~30% of the time** (42/142 items) — 22 items the automated checker missed as pipeline corruption, 20 it over-flagged as corrupted when manual reading judged them a fair test. This led to an explicit, documented practice of hand-reconciling a taxonomy of wrong predictions into categories like *pipeline artifact*, *genuine visual-reasoning miss*, *ambiguous/debatable ground truth*, and *pure resampling noise* — most fully worked out in M5's reconciled breakdown (pipeline artifact 43.7%, genuine miss 34.1%, ambiguous 17.5%, unreviewed carryover 3.2%, resampling noise 1.6%).

### Regression analysis practice

Every milestone from M4 onward reports not just net accuracy change but the **flip counts** underneath it — how many items moved wrong→correct vs. correct→wrong. This surfaced findings a headline number alone would hide: M5's CoT ablation had 32 fixed and 31 regressed for a net of +1 — nearly a wash by count, and only readable as meaningful by also reading the regression traces (which found the reasoning-to-index binding bug). M6 cross-referenced its 27 regressions against M5's own regression sets and found ≥10 items regressed under multiple independent techniques, suggesting a genuinely hard item core rather than pure noise.

### Ablation studies

Every technique evaluated in this project (repair on/off, prompt v1/v2, CoT on/off, few-shot on/off, select model swap) was run as a controlled ablation — **exactly one variable changed at a time**, everything else held fixed, full 500-item split, same seed. This is enforced at the CLI level: `aynvqa-run`'s `--repair-enabled`/`--cot-enabled`/`--fewshot-enabled`/`--ollama-select-model` flags exist specifically to make single-variable ablation runs trivial to invoke correctly.

### Pilot-before-full-scale discipline

Cheap, small-sample pilots (43 items for M5's CoT/few-shot, 5 items for M6's smoke test) are used to catch gross failures cheaply before committing to a ~1-2 hour full-scale run. But this project also explicitly learned the limits of that discipline: M5's CoT pilot recovered 30.2% of hard items, but the full-scale ablation netted only +0.2 points — because CoT touches *every* select call, not just the targeted hard subset, so a pilot restricted to hard items cannot see the regressions CoT introduces elsewhere. The documented lesson: **a pilot on a targeted subset doesn't reliably predict full-scale net effect for any technique that changes behavior on every call, not a targeted subset.**

### How accuracy is measured

`accuracy`, `balanced_accuracy`, and `macro_f1` are all computed by the vendored official scorer and logged verbatim to `experiments.md`. `balanced_accuracy` and `macro_f1` matter alongside raw `accuracy` specifically because of M1's own demonstration: the `constant-0` baseline scores 35.80% accuracy but has a balanced_accuracy of exactly 1/3 (perfect recall on class 0, zero elsewhere) and a much lower macro_f1 of 0.1757 — accuracy alone would make a degenerate always-guess-0 strategy look almost as good as a real one.

### Why changes are validated before adoption

The project's explicit adoption bar, followed consistently from M4 onward: a change is only adopted into the production defaults in `config.py` after a **full 500-item ablation** shows a clear, reproducible improvement — not after a promising pilot alone. This is precisely why CoT and few-shot, despite being fully implemented and passing all unit tests, remain off by default: their pilots looked good, but their full-scale ablations did not clear this bar (+0.2 and +0.0 points respectively), while the M6 select-model swap did (+4.2 points) and was adopted.

---

## 13. Performance history

| Milestone / configuration | Accuracy | N correct / total | Improvement vs. prior | Adopted? |
|---|---:|---:|---:|---|
| `random` selector (M1) | 33.40% | 167/500 | — (chance floor) | Reference only |
| `constant-0` selector (M1) | 35.80% | 179/500 | +2.4 pts vs random | Reference only |
| M3: Whisper-medium → parse → joint-MCQ, no repair | 71.60% | 358/500 | +35.8 pts vs constant | **Adopted** (superseded by M4) |
| M4: + repair ladder, prompt v1 | 73.80% | 369/500 | +2.2 pts vs M3 | Superseded by v2 |
| M4: + repair ladder, prompt **v2** | 74.80% | 374/500 | +1.0 pt vs v1 | **Adopted — FROZEN baseline** |
| M5: + chain-of-thought | 75.00% | 375/500 | +0.2 pt vs M4 frozen | **Rejected** (not adopted; full-scale gain too small relative to complexity, traced to a reasoning-index binding bug) |
| M5: + few-shot (k=2) | 74.80% | 374/500 | +0.0 pt vs M4 frozen | **Rejected** (not adopted; exact wash, 18 fixed / 18 regressed) |
| M6: select model → `qwen3-vl:8b` | 79.00% | 395/500 | +4.2 pts vs M4 frozen | **Adopted — current FROZEN baseline** |

**Progression at a glance:** 33.4% (random) → 35.8% (constant) → 71.60% (M3 cascade) → 73.80% (M4 repair v1) → 74.80% (M4 repair v2, frozen) → 75.00% (M5 CoT, rejected) / 74.80% (M5 few-shot, rejected) → **79.00% (M6 qwen3-vl:8b, adopted, current frozen baseline)**.

All numbers above are cross-checked against `experiments.md`'s raw append-only log to 4 decimal places; no discrepancies were found between the narrative milestone docs and the raw log.

---

## 14. Documentation index

| File | Description |
|---|---|
| `docs/M0_DATA_AUDIT.md` | Design walkthrough for the data-audit milestone: schema/loader/validation/probing/hashing/sampling modules, why each dependency was chosen, and real findings (3 corrupted Git-LFS image stubs, 3 near-duplicate image pairs, 0 exact duplicates). |
| `docs/M1_BASELINES.md` | Design walkthrough for the repo skeleton and two trivial, model-free baselines (random, constant); proves the full load→predict→submit→score→log harness works, with exact chance-floor and label-skew numbers. |
| `docs/M2_ASR_BENCH.md` | ASR-backend selection bench: three implemented backends (Whisper local, Fanar Aura, OpenAI transcribe) but only Whisper actually run; whisper-medium vs whisper-small comparison on a seeded 50-item sample, ending in a "use medium" recommendation for M3. |
| `docs/M3_PIPELINE.md` | The first real end-to-end cascade (Whisper → Ollama-parse → Ollama-joint-MCQ-VLM via local Qwen2.5-VL-7B); full 500-item dev/msa result of 71.60% vs. the official 39.8% MSA baseline, plus per-country/category error breakdown. |
| `docs/M4_PIPELINE_ROBUSTNESS.md` | Diagnoses that ~half of M3's "142 VLM misses" were actually silent parser corruption; builds `option_quality.py` and a 3-step repair escalation ladder, benchmarks it at full scale (71.60%→73.80%→74.80% across baseline/v1/v2 prompts), and freezes 74.80% (374/500) as the new baseline. |
| `docs/M5_FEWSHOT_RETRIEVAL.md` | Evaluates chain-of-thought prompting and category-based few-shot exemplar retrieval against the frozen M4 baseline's genuine visual-reasoning misses; both pilot well (30%/14% recovery on the hard subset) but wash out at full scale (+0.2/+0.0 points); neither adopted. |
| `docs/M6_VLM_SWAP.md` | Swaps the select-stage VLM from qwen2.5vl:7b to qwen3-vl:8b (InternVL ruled out as unavailable in Ollama's official catalog); full-scale run yields +4.2 points (79.00%, 395/500), the largest single-change gain of the project, and is adopted as the new frozen baseline. |

`experiments.md` (project root, not under `docs/`) is the append-only, auto-generated ground-truth log every number above is cross-checked against.

---

## 15. Testing

The test suite is fast, fully offline, and requires no network access, no real model weights, no GPU, and no real multi-gigabyte dataset — everything runs against the synthetic `mini_dataset` fixture defined in `tests/conftest.py`.

### Running tests

```bash
uv run pytest
```
Configured via `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"`.

With coverage (source scoped to `src/ayn_vqa`):
```bash
uv run pytest --cov
```

### Linting

```bash
uv run ruff check .
```
Configured via `[tool.ruff]`: `line-length = 100`, `target-version = "py313"`, rule sets `E, F, I, UP, B, SIM, N`, `extend-exclude = ["scorer_official"]` (vendored code is not linted — "not ours to reformat or fix lint findings in").

### Type-checking

```bash
uv run mypy
```
Configured via `[tool.mypy]`: `python_version = "3.13"`, `strict = true`, `disallow_untyped_defs = true`, `warn_unused_ignores = true`, `files = ["src", "tests"]`. `soundfile.*` and `faster_whisper.*` are given `ignore_missing_imports = true` (no upstream type stubs exist for either).

### Expected output

A clean run of all three commands against the current codebase (23 test files, 2,278 lines, as of M6) should report all tests passing, zero ruff findings, and zero mypy errors — these three checks are run before every milestone's results are considered final, per the development workflow in §16.

### Test suite composition (highlights)

- `test_run_pipeline.py` (447 lines) — the largest file; exercises the full M3+ cascade end to end with fake stages, calling `run_pipeline()` directly with explicit kwargs (`repair_enabled=True`, `fewshot_enabled=True`, `fewshot_train_records=...`, etc.) rather than through the CLI/argparse layer.
- `test_select_vlm.py`, `test_option_quality.py`, `test_retrieve.py`, `test_parse.py` — cover the M3–M5 stage logic.
- `test_ollama_client.py` — verifies actual HTTP request shape against `httpx.MockTransport`, never a live server.
- `test_asr.py` — `WhisperLocalASR` via an injected fake model; `FanarAuraASR`/`OpenAITranscribeASR` via `httpx.MockTransport`.
- No CI configuration exists in this repository (no `.github/workflows/`) — all of the above is run manually as part of the development workflow described next.

---

## 16. Development workflow

This project follows one consistent methodology across all six milestones, visible directly in the structure of every `docs/M*.md` file and in `experiments.md`'s ablation-run pattern:

1. **Implement.** Write the stage/module/flag in `src/ayn_vqa/`.
2. **Unit tests.** Add or extend tests against the synthetic `mini_dataset` fixture — no real dataset, no live model calls, no network.
3. **`ruff check .`** — lint clean (excluding vendored `scorer_official/`).
4. **`mypy`** — strict type-check clean.
5. **Small pilot.** Run against a cheap, seeded sample (43 items for M5's CoT/few-shot pilots, 5 items for M6's smoke test, 50 items for most `--sample-n` sanity checks) to catch gross failures before committing to a long full-scale run.
6. **Full benchmark.** Run the complete 500-item `dev`/`msa` split, scored by the vendored official scorer, one variable changed at a time versus the current frozen baseline.
7. **Regression analysis.** Don't stop at the net accuracy delta — count flips (wrong→correct vs. correct→wrong) and read a sample of the actual traces. This is how M5 caught the reasoning-to-answer-index binding bug that a bare "+1 net" number would have hidden, and how M6 connected its regressions back to M5's own regression sets.
8. **Documentation.** Write up the milestone in `docs/M<N>_<NAME>.md` — goal, implementation, exact numbers, lessons learned, design decisions, explicit adoption/rejection status.
9. **Adopt or reject.** Only promote a change into `config.py`'s baked-in defaults if the full-scale ablation clears a real bar (a clear, reproducible improvement) — not merely a promising pilot. This is why CoT and few-shot remain implemented, tested, and off by default, while the M6 select-model swap is on by default.

This is also why the project was able to catch the M6 cache-collision bug (§9) *before* wasting a ~2-hour run: step 3-4 (lint/type-check) plus deliberate code review of the caching mechanism, informed directly by the earlier parser v1→v2 lesson, caught it at implementation time rather than after the fact.

---

## 17. Future work

Grounded in explicit statements in `docs/M5_FEWSHOT_RETRIEVAL.md` and `docs/M6_VLM_SWAP.md` — nothing below is speculative beyond what those docs themselves flag as open:

- **CoT v2.** M5 identified two specific, fixable bugs in the current CoT implementation — a reasoning-to-answer-index reliability gap (≥7 clear cases where the free-text `reasoning` field names the correct option but the constrained-decoded `answer_index` doesn't match) and a "fabricated-placeholder seduction" pattern (≥4 cases of careful reasoning about, and selecting, a known-degenerate option). A cheap fix — an explicit "reasoning must match index" instruction — was named as a plausible fast follow but deliberately not implemented in M5, since CoT wasn't adopted as-is. Whether CoT is worth re-evaluating against the new M6 baseline (79.00%) is explicitly called out in `docs/M6_VLM_SWAP.md` as an open question, not answered there.
- **Few-shot classifier improvements.** M5 diagnosed the real bottleneck as category-classification precision (a single 9-way call from one image), not the retrieval mechanism itself — e.g. "type of car shown" was misclassified into *Culture, Arts & Entertainment* instead of *Vehicles & Transportation*. Improving that single classification call, rather than the exemplar retrieval or prompting around it, is the most direct lever for few-shot re-evaluation. Also explicitly named as open against the new M6 baseline.
- **Ensemble models.** M6's original plan included an ensemble step, deliberately not executed: the project's own adoption criterion ("no single model consistently better, so ensemble") wasn't met — qwen3-vl:8b's +4.2-point gain was clear and consistent enough that a single-model swap was adopted directly instead.
- **OCR.** M4's root-cause analysis of the 37 repair-unresolved cases found that 16/37 (43%) were dataset issues where the transcript honestly contains only 2 spoken options and the third piece of information needed is visible only as on-image text — an OCR channel was named as the fix for this subset but not built.
- **Fine-tuning.** Not implemented in any milestone — the entire project stays on off-the-shelf Whisper and Ollama-served VLMs with prompt-level and pipeline-level interventions only; no training/fine-tuning code exists anywhere in `src/`.
- **New VLMs.** M6 evaluated exactly one alternative (Qwen3-VL:8b) against exactly one ruled-out candidate (InternVL, unavailable as a first-party Ollama model). Whether other first-party Ollama VLMs would move the needle further is unexplored and not scheduled.
- **Question-text-overlap check for fabricated options.** Proposed in M4 (flag an option sharing an unusually long substring with the question text) as a sharper detector than the current `option_quality.py` heuristics, but not implemented.
- **Parser prompt/model cache-key hardening.** Not user-facing "future work" in the docs, but a real, load-bearing engineering gap noted in §9: `OllamaTranscriptParser.name` still doesn't vary with prompt text or model, unlike every other stage. A future change could hash the prompt template into the cache key to close this gap automatically.

---

## 18. FAQ / Troubleshooting

**"`cublas64_12.dll is not found`" when running Whisper on GPU.**
*Cause:* `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` ship their DLLs inside the pip wheel rather than a system-wide CUDA install. PyTorch auto-registers these wheel-local DLL directories at import time; `ctranslate2` (what `faster-whisper` actually runs inference on) does not do this itself, so on Windows the failure only surfaces at the *first* `.transcribe()` call, not at model construction. *Fix (already applied in this repo):* `stages/asr.py`'s `_register_windows_cuda_dll_dirs()` calls `os.add_dll_directory()` on the wheel's bundled DLL directories, gated to `device.startswith("cuda")` and Windows only. If you still see this error, confirm the `gpu` dependency group is actually installed (`uv sync --group gpu`) and that `AYNVQA_WHISPER_DEVICE`/`AYNVQA_REPAIR_WHISPER_DEVICE` are set to `cuda`.

**Whisper large-v3 seems like it should be slower than medium — is GPU actually worth it?**
No — measured directly in M4: large-v3 on GPU (float16) ran at **1.32s mean latency** vs. whisper-medium's **7.92s on CPU** — roughly 6x *faster*, with near-identical transcript length (115 vs 114 mean chars). Once the CUDA DLL issue above is resolved, escalating to a stronger Whisper model at the repair stage is essentially free.

**Ollama times out ("400" error) on multi-image (few-shot) prompts.**
*Cause:* Ollama's own default `num_ctx` is 4096 tokens, confirmed by direct testing to time out on more than one image in a single request. *Fix (already applied in this repo):* the `fewshot_num_ctx` setting (default `16384`) is applied specifically to the multi-turn few-shot call path via `OllamaClient.chat_messages`. If you build a new multi-image call path yourself, make sure to pass a similarly raised `num_ctx`.

**Why does the select/parse stage sometimes just return a fallback prediction with no model call?**
By design. If the parse stage failed or has no usable options, the select stage returns `Prediction(FALLBACK_INDEX, ...)` directly, bypassing both the model call and the cache — cheap, and self-healing: once the upstream ASR/parse problem is fixed (or the cache cleared), the record will naturally be re-attempted on the next run with no manual intervention.

**I changed the parse prompt (or `AYNVQA_OLLAMA_PARSE_MODEL`) and my results didn't change at all.**
This is the parser cache-key gap described in §9: `OllamaTranscriptParser.name` is the hardcoded constant `"ollama-parse"` and does **not** vary with model or prompt text, so your edit is being silently served from the old cache. *Fix:* manually delete or archive `artifacts/<split>/parse/*_ollama-parse.jsonl` before re-running. This is exactly what happened during the real M4 v1→v2 prompt swap, and was resolved the same way (archived into `artifacts/_v1_prompt_archive/`).

**I swapped `--ollama-select-model` and the results look suspiciously identical to the old model.**
Check whether you're on a version of this repo that predates the M6 fix — before it, `OllamaJointMCQSelector.name` varied with the CoT flag but never with the selected model, so a model swap could silently collide with and replay the default model's existing select-stage cache. This was fixed and is covered by two tests as of M6; if you still see this, verify `selector.name` (visible in the select-stage cache filename) actually includes your model's slug, e.g. `ollama-joint-mcq-qwen3-vl-8b`.

**Structured JSON parsing keeps failing / the model returns malformed output.**
This project deliberately avoids free-text-plus-regex parsing (the official baseline's own `re.search(r"[012]", ...)` approach, which silently falls back to index 0 on failure) in favor of Ollama's JSON-schema-constrained decoding for both the parse stage (`{question, option_0, option_1, option_2}`) and the select stage (`answer_index` constrained to `{0,1,2}`). If you're extending this pipeline with a new stage or a new model, use the same `format=<json_schema>` mechanism in `ollama_client.py` rather than free-text parsing — schema violations were the specific failure mode M6's pre-run smoke test (5 items) checked for before committing to a full run.

**A model I want to use isn't available in Ollama.**
Verify directly against Ollama's official model catalog before planning around it — this project found InternVL specifically was **not available** as a first-party Ollama model (only unofficial community GGUF conversions exist, e.g. `blaifa/InternVL3`, with no first-party QA and unverified compatibility with schema-constrained decoding) and ruled it out on that basis in M6. Don't assume a model exists in Ollama's catalog just because it's a well-known open-weight VLM.

**GPU memory / hardware — how much VRAM do I need?**
This project's GPU is an NVIDIA RTX 2000 Ada Generation with 16GB VRAM (confirmed via `nvidia-smi` in M6, correcting an earlier informal reference to an RTX 4080 — same VRAM capacity, different/lower throughput card). Qwen3-VL:8b (6.14GB) and Qwen2.5-VL:7b (5.97GB) both fit comfortably alongside Whisper large-v3 on a 16GB card; no OOM issues have been reported in this project's own runs.

**Console output shows garbled Arabic text / crashes with `UnicodeEncodeError` on Windows.**
*Cause:* Rich's legacy console renderer previously fell back to writing through the system ANSI codepage (`cp1252`), which cannot encode Arabic — the underlying cache write was always safe (explicit UTF-8 file I/O), but console output was unreadable stack traces. *Fix (already applied in this repo):* `logging_utils.py` forces `legacy_windows=False` and reconfigures `stdout`/`stderr` to UTF-8. If you still see this on a fresh environment, check that nothing else in your setup is overriding console encoding after `logging_utils` runs.

**Model downloads — how big are the Ollama pulls, and how long do full runs take?**
`qwen2.5vl:7b` is ~5.97GB, `qwen3-vl:8b` is ~6.14GB. Full 500-item `dev`/`msa` runs have taken roughly: ~2h4m (M3, first cascade, whisper-medium-CPU-dominated), ~56min (M4 repair v1), ~63min (M4 repair v2), and ~2h14m (M6, qwen3-vl:8b — the longest so far, reflecting the model's markedly higher and more variable latency: 7.8s–71.7s observed in the M6 smoke sample, vs. qwen2.5vl's typical ~4-5s). A cached re-run of an identical configuration completes in seconds regardless of which model was used, since every stage's output is cached (see §9).

**Slow inference in general — what's the biggest lever?**
For ASR, GPU (once the CUDA DLL fix is in effect) is dramatically faster than CPU (§ above). For the select stage, model choice dominates latency more than accuracy might suggest — qwen3-vl:8b's higher/more variable latency was flagged explicitly *before* the M6 full run as an expected tradeoff for its accuracy gain, not a surprise discovered afterward.