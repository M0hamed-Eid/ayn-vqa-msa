# M0 -- Data Audit: design walkthrough

This is the "why", file by file, for the M0 milestone: understand the raw
Task 1a data before any ASR/OCR/VLM code is written. If you're picking this
project back up later, read this before touching `src/ayn_vqa/`.

Roadmap context lives in
[`../../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
(§11, M0). This doc is the implementation-level companion to that plan.

## Why a separate project (not inside the dataset clone)

`AynVQA-ArabicNLP26/` (the sibling folder) is a git-LFS clone whose `origin`
is the QCRI HuggingFace **dataset** repo -- not something we have push
access to, and not the right place for pipeline code and its own commit
history. This project (`ayn-vqa-msa/`) is a separate git repo that
*references* that data by path (`AYNVQA_DATA_ROOT` in `.env`) and never
copies or modifies it. If the dataset ever moves, or you run this on a
different machine, one env var changes and nothing else does.

## `pyproject.toml`

Declares the package (`ayn-vqa`, `src/` layout via `uv init --package`),
runtime dependencies, three dev/notebook dependency groups, and tool
config for `ruff`, `mypy` (strict), and `pytest` in one file instead of
scattering `setup.cfg`/`mypy.ini`/`pytest.ini`. `uv add`/`uv add --group`
keep this file's dependency lists in sync automatically -- it's never
hand-edited for a version bump.

Why these specific runtime dependencies:

| Package | Why this one |
|---|---|
| `pydantic` / `pydantic-settings` | One schema for the JSONL records (`Task1aRecord`) and one schema for config (`Settings`), both with real validation instead of raw dicts. |
| `soundfile` | Reads WAV headers via libsndfile without decoding full sample buffers, and (unlike stdlib `wave`) doesn't choke on non-PCM subtypes. |
| `pillow` | Same lazy-header idea for images, plus first-class GIF/animation support. |
| `pandas` | Turns per-record stat dataclasses into groupby/describe/value_counts in a few lines instead of hand-rolled aggregation. |
| `matplotlib` | The handful of static PNG figures embedded in the Markdown report and notebook. |
| `rich` | Readable, leveled console logging instead of bare `print`. |
| `tabulate` | Backs `DataFrame.to_markdown()` for the report's tables. |

Deliberately *not* a dependency: `imagehash` (adds `scipy` + `PyWavelets`
transitively for one hash function we implement in ~15 lines in
`hashing.py`), and any CLI framework (`click`/`typer`) for what is one
`argparse` command.

## `.env.example` / `src/ayn_vqa/config.py`

Every path, seed, and threshold is a `Settings` field read from
`AYNVQA_*` env vars (via `pydantic-settings`), never hardcoded at the call
site. `get_settings()` is a factory, not a module-level singleton --
tests construct their own `Settings(...)` without fighting import-time
caching. Relative paths (like the default `../AynVQA-ArabicNLP26`) resolve
against the *project root*, computed from `Path(__file__).resolve()`, not
`Path.cwd()` -- so the CLI, a notebook, and pytest all behave identically
regardless of where they're launched from.

## `src/ayn_vqa/logging_utils.py`

One `setup_logging()` call configures the root logger with a Rich console
handler (readable, leveled) and an optional file handler. No other module
calls `logging.basicConfig` or uses bare `print` for anything but final
human-facing summaries -- that's what makes "run this quietly" or "log to
a file instead" a one-line change instead of a grep-and-replace.

## `src/ayn_vqa/data/schema.py`

`Task1aRecord` is a frozen, `extra="forbid"` Pydantic model matching the
exact key sets verified against all six real JSONL files during this
milestone (`train`/`dev` have `label`/`country`/`category`/`subcategory`;
`devtest` has only `id`/`image`/`audio`). `extra="forbid"` is intentional:
if the eventual blind `test` split introduces a surprise field, loading it
should fail loudly here, not silently drop data three stages downstream.
`label` is validated to be exactly `0`, `1`, or `2` when present. Every
later stage (ASR, parsing, VLM, in M1+) imports this one model instead of
re-parsing `dict["image"]` by hand.

## `src/ayn_vqa/data/loader.py`

Parses `task1a/<split>_<language>.jsonl` line by line with a plain `for`
loop and `json.loads` per line -- not `pandas.read_json(lines=True)` or
`datasets.load_dataset`, both of which raise on (or bury the location of)
the first malformed line. `load_split` returns the records that parsed
*and* a list of `RecordError` (line number, raw text, message) for the
ones that didn't, because a data-audit tool that crashes on line 1 of a
3,000-line file defeats its own purpose.

## `src/ayn_vqa/data/validation.py`

For every record, checks whether `image` and `audio` resolve to a real
file (`MediaCheck`: exists, size). This is the check M0 exists to run:
surfacing "12 files missing, here they are" before any model call, instead
of a cascade three milestones from now silently skipping or crashing on a
missing file.

## `src/ayn_vqa/audit/audio_stats.py` and `image_stats.py`

Both follow the same shape: a `probe_*(record_id, path)` function that
*never raises* -- a missing file or a corrupt/undecodable file becomes an
`error` field on the returned dataclass, not an exception. `soundfile.info`
and `PIL.Image.open` both do lazy/header-only reads, so probing thousands
of files is fast. `image_stats` explicitly surfaces `is_animated`/
`n_frames` for GIFs rather than silently treating them as a single static
frame.

**Real finding from running this against the actual dataset:** 3 of the
4,000 MSA-track images raise `PIL.UnidentifiedImageError`. Inspecting the
raw bytes shows they're un-smudged **Git LFS pointer stubs** (134-byte
text files: `version https://git-lfs.github.com/spec/v1` / `oid sha256:...`
/ `size ...`), not real image data -- `git lfs pull` on the dataset clone
never fully materialized them. This is exactly the class of silent failure
this milestone exists to catch. Fix: from `AynVQA-ArabicNLP26/`, run
`git lfs pull --include "images/*"` (or `git lfs fetch --all && git lfs checkout`)
to re-download the missing blobs -- this only fills in already-tracked LFS
objects, it doesn't rewrite history or touch anything else.

## `src/ayn_vqa/audit/hashing.py`

Two independent duplicate checks, because they catch different bugs:

- **Exact** -- SHA-256 of raw file bytes. Catches the same file copied
  under two ids. Streamed in 1 MiB chunks so large images don't need to
  fit in memory at once.
- **Near** -- a difference hash (dHash): shrink to a 9x8 grayscale grid,
  compare each pixel to its right neighbor, pack the 64 booleans into an
  int. Catches the same *picture* re-encoded/resized/re-compressed under
  two ids, which changes every byte (defeating SHA-256) but barely moves
  the fingerprint. Needs only Pillow -- no `imagehash`/`scipy` dependency.

`find_near_duplicates` is a plain O(n²) all-pairs Hamming-distance
comparison. At ~4,000 images that's a few million `int.bit_count()` calls
-- a couple of seconds in CPython. An indexed structure (BK-tree, LSH)
would be the right call at 10-100x this scale; building one now would be
premature engineering for data this small.

**Real finding:** 0 exact duplicates, 3 near-duplicate pairs across the
full MSA track (see `reports/m0_data_audit/duplicate_images.csv` for the
ids) -- a real answer to a risk the project plan had flagged as
"⚠️ not yet checked."

## `src/ayn_vqa/audit/sampling.py`

`sample_records` uses `random.Random(seed).sample`, **never** `records[:n]`
-- the real `train`/`dev` JSONL files are sorted alphabetically by
country, so a slice would silently sample one country only.
`render_sample_grid` builds a contact sheet (thumbnail + id/country/label
caption per cell) and renders a labeled placeholder tile instead of
raising when an image is missing or corrupt -- consistent with this
package's rule that broken data should become visible, not crash the tool
that's supposed to find it.

## `src/ayn_vqa/audit/report.py`

The aggregation layer: folds every dataclass the other modules produced
into a `pandas.DataFrame` (for `file_manifest.csv`/`duplicate_images.csv`)
and a plain nested `dict` (for `audit_summary.json`/`audit_report.md`).
It holds no probing logic itself -- that split is what makes each half
independently testable (fake a DataFrame vs. fake a WAV file).

Two non-obvious details worth knowing if you touch this file:

- `_to_native` recursively converts pandas/numpy output (numpy scalar
  types, NaN, non-`str` dict keys) into plain `json.dumps`-safe Python.
  numpy `int64`/`bool_` don't subclass the builtin `int`/`bool`, and
  `json.dumps` rejects non-`str`/`int`/`float`/`bool`/`None` dict keys --
  both are exactly the kind of thing that works in a quick test and then
  throws `TypeError` the first time a real column happens to contain them.
- The `label` column is cast to pandas' nullable `"Int64"` dtype rather
  than left as plain `int`/`float64`. Mixing Python `None` (from
  unlabeled `devtest` rows) into a numeric column upcasts it to `float64`,
  which would print labels as `"0.0"`/`"1.0"`/`"2.0"` in the report instead
  of `"0"`/`"1"`/`"2"`.

## `src/ayn_vqa/audit/run_audit.py`

The orchestrator and `aynvqa-audit` CLI entry point. Wires the modules
above into one pass per split: load → validate → probe audio → probe
image → (after all splits) hash every image → sample → write reports. It
contains no probing/aggregation logic of its own -- only argument parsing,
sequencing, and console logging -- so it stays easy to read even though
it's the file that "does the most."

## `tests/`

`conftest.py`'s `mini_dataset` fixture builds a complete synthetic dataset
(images, WAVs, JSONL splits, matching the real schema exactly) in code
under `tmp_path`, rather than checking in binary fixture files. That means
it can never drift out of sync with `Task1aRecord`, runs in milliseconds,
and needs no network/LFS access -- the entire suite (30 tests) runs in
under 2 seconds. It deliberately encodes three "bad data" cases the real
dataset also has some version of: a malformed JSON line, a record whose
audio file was never created, and a byte-identical duplicate image under a
different id. Coverage is unit-level per module (`test_schema.py` through
`test_hashing.py`) plus one end-to-end integration test
(`test_run_audit.py`) that proves the modules actually compose, not just
that each one works in isolation.

## `notebooks/00_data_audit.ipynb`

A runnable walkthrough of sections 1-7 above (loads `dev` only for
speed in the exploratory steps, then runs the full `train`+`dev`+`devtest`
audit in the final section -- identical to `uv run aynvqa-audit`). Executed
in place (`jupyter nbconvert --execute --inplace`) so it ships with real
output already baked in; re-run it after any code change to keep it that
way, since a notebook with stale outputs is worse than no notebook.

## What's deliberately not here

No ASR, no OCR, no transcript parsing, no VLM calls, no `stages/` package.
Per the roadmap, that's M1 (repo skeleton + trivial baselines) onward.
This milestone's only job was to answer "what exactly is in this data,
and is any of it broken" before a single model call is made against it.
