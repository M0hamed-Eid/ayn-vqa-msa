# Task 1 Format Checker

Run `check_format.py` before zipping your predictions as `prediction.zip` and
uploading to Codabench. It catches the common mistakes early so you don't waste a
submission.

```bash
# Task 1a: one row per item: id,prediction (0/1/2)
python check_format.py --task 1a --pred prediction.csv

# Task 1b: three rows per item: id,statement_index,prediction (true/false)
python check_format.py --task 1b --pred prediction.csv

# optionally check that your ids exactly match a split
python check_format.py --task 1a --pred prediction.csv --gold ../data/task1a/devtest_en.jsonl
```

The checker verifies:

- **Task 1a**, columns `id,prediction`; every `prediction` is a single integer
  `0`, `1` or `2`; no duplicate ids.
- **Task 1b**, columns `id,statement_index,prediction`; `statement_index` ∈
  {0,1,2}; exactly the three statements present per item; no duplicate
  `(id, statement_index)`; it **warns** about verdicts the scorer's `evaluate_tf`
  parser can't read (those count as `false`).
- **With `--gold`**, your ids exactly match the split (no missing/extra).

Exit code `0` = ready to submit, `1` = fix the reported errors first.
