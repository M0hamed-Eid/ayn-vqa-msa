# Vendored: official scorer + format checker

Copied **unchanged** from
[`ImageEval2026/ImageEval2026-tasks`](https://github.com/ImageEval2026/ImageEval2026-tasks),
commit `791b596222cbfec40dd8a57102adb81a8b75116d` (2026-06-22).

```
scorer_official/
├── scorer/{score.py, backbone.py, README.md}
└── format_checker/{check_format.py, README.md}
```

**Do not edit these files.** The whole point of vendoring is that our local
dev-score can never silently drift from what the Codabench leaderboard
actually computes. If the organizers update the scorer, re-copy it from the
source repo (update the commit hash above) rather than patching it here.

`backbone.py` (the `evaluate_tf` True/False parser) is only used by Task
1b/1c scoring -- irrelevant to this project's Task 1a MSA track -- but is
vendored alongside `score.py` because `score.py` imports it at module load
time; keeping the original directory layout means the vendored files need
zero modification to run.

`src/ayn_vqa/evaluate.py` and `src/ayn_vqa/submission.py` import
`score_1a`/`check_1a` directly from these files (via `sys.path`, resolved
relative to the project root) rather than reimplementing the metrics --
see `docs/M1_BASELINES.md` for why.
