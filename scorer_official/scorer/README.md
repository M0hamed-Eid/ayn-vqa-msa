# Task 1 Scorer

`score.py` computes **exactly the Codabench leaderboard metrics**, so you can
evaluate yourself locally on a labelled split (e.g. `dev`) before submitting.
Gold labels are read straight from the released JSONL (`label` for 1a, `labels`
for 1b). A missing or unparseable prediction always counts as wrong.

```bash
python score.py --task 1a --gold ../data/task1a/dev_en.jsonl --pred prediction.csv
python score.py --task 1b --gold ../data/task1b/dev_en.jsonl --pred prediction.csv
```

The first metric printed (marked `<-- RANKING`) is the one that orders the
leaderboard. Pure Python standard library, no extra dependencies.

## Task 1a: Spoken VQA

| metric | role | meaning |
|---|---|---|
| **accuracy** | **ranking** | fraction of items answered correctly |
| balanced_accuracy | reported | mean per-class recall over the three option positions |
| macro_f1 | reported | macro-averaged F1 over the three option positions |

## Task 1b: Hallucination Detection

Exactly one statement per item is grounded ("Q+"); the other two are
hallucinated ("Q−").

| metric | role | meaning |
|---|---|---|
| **contrastive_instability** | **ranking** | of items with ≥1 of the three labels correct, the fraction not fully correct (all three right); lower is better |
| combined_accuracy | reported | fraction of items where **all three** labels are correct |
| cfhr | reported | of items whose grounded statement was correctly identified, the fraction that still affirmed a hallucinated one; lower is better |
| q_plus_accuracy | reported | grounded statement correctly marked true |
| q_minus_accuracy | reported | hallucinated statements correctly marked false (over all false statements) |

True/False is read with the shared-task `evaluate_tf` parser (in
[`backbone.py`](./backbone.py), bundled here so this scorer is identical to the
Codabench one). It handles English and Arabic verdicts: `true`/`false`, `yes`/`no`,
صح/خطأ, etc.
