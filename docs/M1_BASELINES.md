# M1 -- Repo skeleton + trivial baselines: design walkthrough

The "why", file by file, for M1: prove the whole harness -- load a split,
predict something, write a submission, check its format, score it, and
remember the score -- works end to end before any real model (ASR, VLM)
exists to plug into it. Companion to
[`docs/M0_DATA_AUDIT.md`](M0_DATA_AUDIT.md) and
[`../../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
(§11, M1).

## `scorer_official/`

Copied **unchanged** from the organizers' `ImageEval2026/ImageEval2026-tasks`
repo (see `scorer_official/README.md` for the exact commit). This is
deliberate: our local dev score must never be *our* interpretation of
"accuracy" -- it has to be the organizers' exact `score_1a`, imported
directly, so it can never silently drift from what the Codabench
leaderboard computes. `backbone.py` is vendored alongside `score.py` only
because `score.py` imports it at load time (it's the Task 1b/1c
True/False parser -- irrelevant to our Task 1a MSA work, but keeping the
original directory layout means the vendored files need zero
modification to run). Excluded from `ruff`'s lint scope in `pyproject.toml`
-- it's not ours to reformat or fix findings in.

## `src/ayn_vqa/stages/select.py`

`AnswerSelector` is a `Protocol`, not a base class: M2+'s real selectors
(ASR + joint-MCQ VLM prompting, entailment ranking, the end-to-end omni
model) will implement the same one-method shape (`predict(record) ->
Prediction`) without inheriting from anything or this file needing to
change. Two selectors exist for M1, and both are deliberately model-free:

- `RandomSelector` -- a seeded uniform guess over `{0,1,2}`. This is the
  dataset's chance floor (~33%, since the label distribution is close to
  balanced -- see `docs/M0_DATA_AUDIT.md`).
- `ConstantSelector` -- always predicts the same index. This exposes
  exactly how much accuracy comes "for free" from the label
  distribution's mild skew (dev is 35.8%/31.6%/32.6% -- so
  `ConstantSelector(0)` scores ~36%) rather than from actually looking at
  the image or audio. Any real selector built in M3+ that can't beat both
  numbers isn't using its inputs at all.

`predict_split` is the harness: any `AnswerSelector` over any list of
records, in order.

## `src/ayn_vqa/submission.py`

`write_predictions_csv` writes exactly the `id,prediction` columns the
scorer and Codabench expect. `check_submission_format` then runs the
*vendored* `check_1a` against that file -- not our own belief that the
writer is correct. Catching a formatting bug locally costs nothing;
catching the same bug via a failed Codabench upload costs one of a small
number of daily submission attempts.

## `src/ayn_vqa/evaluate.py`

Wraps the vendored `score_1a`. Two non-obvious decisions here, both found
by actually running the tests, not by inspection:

- **Takes `list[Task1aRecord]`, not a gold JSONL path.** An earlier
  version re-read the raw JSONL file directly and broke immediately: the
  M0 test fixture's `train_msa.jsonl` deliberately contains one malformed
  line (to test the *loader's* tolerance), and a second, naive JSON parser
  reading the same file has no such tolerance. Scoring now always goes
  through records `data/loader.py` already parsed and validated, so there
  is exactly one JSONL parser in this codebase, not two that can silently
  disagree.
- **Rejects unlabeled gold before calling the vendored code, not after.**
  The vendored `score_1a` only checks that a `"label"` *key* exists, not
  that its value is non-`None`. Passing `devtest`-style records straight
  through would make every item's gold label the *string* `"None"`
  instead of raising -- a silent correctness bug, not a crash. `evaluate.py`
  checks `any(r.label is None ...)` itself first and raises a clear
  `RuntimeError`. (`score_1a` also still calls `sys.exit(1)` on some other
  malformed input, e.g. a prediction CSV missing its columns -- fine for a
  standalone CLI script, but that would kill our whole process, so that
  path is still converted to `RuntimeError` too.)

## `src/ayn_vqa/experiments.py`

`reports/` is regenerated (and gitignored) by every run of `aynvqa-audit`
or `aynvqa-predict` -- the wrong place to remember "what did we already
try, and how did it score" across sessions. `experiments.md` lives at the
project root, is checked into git, and only ever grows: one Markdown table
row per scored run. Scoped to Task 1a's fixed 3-metric shape for now; a
later milestone scoring a different task would need a per-task header
rather than one shared table -- not a concern until that milestone exists.

## `src/ayn_vqa/run_baseline.py`

The `aynvqa-predict` CLI entry point and the M1 harness itself: load ->
predict -> write CSV -> format-check -> (if labeled) score -> log to
`experiments.md`. `run_baseline()` (explicit parameters) is kept separate
from `main()` (argparse glue), the same split used in
`audit/run_audit.py`, so tests call the former directly without going
through `sys.argv`.

## Real results (MSA track, `dev`, 500 items)

```
random:      accuracy 0.3340  balanced_accuracy 0.3322  macro_f1 0.3320
constant-0:  accuracy 0.3580  balanced_accuracy 0.3333  macro_f1 0.1757
```

Matches the roadmap's expected "~33%/~36%" almost exactly. Worth noticing:
`constant-0`'s balanced_accuracy is *exactly* 1/3 (perfect recall on class
0, zero recall on classes 1 and 2 -- `(1+0+0)/3`), while its macro-F1 is
much lower (0.1757) because precision on class 0 is only ~0.358. This is
the concrete difference between "accuracy" and "balanced accuracy vs.
macro-F1" that M4's position-bias work will come back to.

## Tests

`test_select.py`'s first draft had a real bug worth remembering: it built
a fresh `RandomSelector(seed=N)` *inside* the list comprehension, once per
record, so every "run" was actually 50 copies of the RNG's first draw --
the assertion passed, but for the wrong reason (both runs were trivially
constant, not a real reproduced sequence). Fixed by constructing one
selector instance per run and calling `.predict()` on it repeatedly, which
is also how `predict_split` actually uses it. `test_evaluate.py` is the
project's explicit M1 acceptance test: 10 hand-labeled items, a prediction
with a specific wrong-answer pattern, and hand-computed
accuracy/balanced-accuracy/macro-F1 (`0.7`, `25/36`, `44/63`) asserted
against what `score_predictions` actually returns.

## What's deliberately not here

No ASR, no OCR, no VLM prompting, no config-driven experiment YAML system.
The plan's full architecture (§10) wants `configs/` for swapping models by
config file -- premature before M2 introduces the first real model choice
(which ASR system) to swap between. Two trivial, hardcoded-by-CLI-flag
baselines are all M1 asked for; `configs/` arrives when there's an actual
second axis of variation to configure.
