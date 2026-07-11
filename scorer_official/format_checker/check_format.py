"""
Ayn-VQA Submission Format Checker (Task 1a / Task 1b)
=====================================================
Validates a prediction CSV before you zip it as ``prediction.zip`` and upload to Codabench. Catches the common mistakes early so you don't waste a submission.

Usage
-----
    # Task 1a — one row per item: id,prediction  (prediction in {0,1,2})
    python check_format.py --task 1a --pred prediction.csv

    # Task 1b — three rows per item: id,statement_index,prediction (true/false)
    python check_format.py --task 1b --pred prediction.csv

Optionally pass --gold to also check that your ids exactly match a split:
    python check_format.py --task 1a --pred prediction.csv --gold ../data/task1a/devtest_en.jsonl

The Codabench scorer reads True/False with the same shared-task ``evaluate_tf`` parser (English and Arabic verdicts). Predictions it cannot parse count as
``false``; this checker warns you about those rows so nothing is silently lost.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scorer"))
try:
    from backbone import evaluate_tf  # used only to warn about unparseable 1b verdicts
except Exception:                     # checker still works without it
    evaluate_tf = None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh)), (csv.DictReader(open(path, encoding="utf-8-sig")).fieldnames)


def load_gold_ids(gold_path):
    ids = []
    with open(gold_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.append(str(json.loads(line)["id"]).strip())
    return ids


# ────────────────────────────────────────────────────────────────────────────
def check_1a(rows, fieldnames, gold_ids):
    errors, warnings = [], []
    required = ["id", "prediction"]
    missing = [c for c in required if c not in (fieldnames or [])]
    if missing:
        errors.append(f"Missing required columns: {missing}. Found: {list(fieldnames or [])}")
        return errors, warnings

    seen = {}
    for i, row in enumerate(rows, start=2):
        iid = (row.get("id") or "").strip()
        pred = (row.get("prediction") or "").strip()
        if not iid:
            errors.append(f"Row {i}: missing id.")
            continue
        if iid in seen:
            errors.append(f"Row {i}: duplicate id '{iid}' (also row {seen[iid]}).")
        seen[iid] = i
        if pred not in {"0", "1", "2"}:
            errors.append(f"Row {i} ({iid}): prediction = '{pred}' must be a single integer 0, 1 or 2.")

    _check_ids(seen.keys(), gold_ids, errors, warnings)
    return errors, warnings


def check_1b(rows, fieldnames, gold_ids):
    errors, warnings = [], []
    required = ["id", "statement_index", "prediction"]
    missing = [c for c in required if c not in (fieldnames or [])]
    if missing:
        errors.append(f"Missing required columns: {missing}. Found: {list(fieldnames or [])}")
        return errors, warnings

    seen_pairs = {}
    per_item = {}          # id -> set of statement_index seen
    unparseable = 0
    for i, row in enumerate(rows, start=2):
        iid = (row.get("id") or "").strip()
        si_raw = (row.get("statement_index") or "").strip()
        pred = row.get("prediction") or ""
        if not iid:
            errors.append(f"Row {i}: missing id.")
            continue
        try:
            si = int(si_raw)
        except ValueError:
            errors.append(f"Row {i} ({iid}): statement_index = '{si_raw}' must be 0, 1 or 2.")
            continue
        if si not in {0, 1, 2}:
            errors.append(f"Row {i} ({iid}): statement_index = {si} must be 0, 1 or 2.")
            continue
        key = (iid, si)
        if key in seen_pairs:
            errors.append(f"Row {i}: duplicate (id, statement_index) = {key} (also row {seen_pairs[key]}).")
        seen_pairs[key] = i
        per_item.setdefault(iid, set()).add(si)
        if evaluate_tf is not None and evaluate_tf(pred).pred is None:
            unparseable += 1

    # every item needs exactly the three statement indices
    for iid, idxs in per_item.items():
        miss = {0, 1, 2} - idxs
        if miss:
            errors.append(f"Item '{iid}': missing statement_index {sorted(miss)} (need rows for 0, 1 and 2).")

    if unparseable:
        warnings.append(f"{unparseable} prediction(s) could not be parsed as true/false and will count as 'false'. "
                        "Use plain 'true'/'false' (or Arabic صح/خطأ).")

    _check_ids(per_item.keys(), gold_ids, errors, warnings)
    return errors, warnings


def _check_ids(pred_ids, gold_ids, errors, warnings):
    if gold_ids is None:
        return
    pred_set, gold_set = set(pred_ids), set(gold_ids)
    missing = gold_set - pred_set
    extra = pred_set - gold_set
    if missing:
        errors.append(f"{len(missing)} id(s) from the gold split are missing from your predictions "
                      f"(e.g. {sorted(missing)[:3]}).")
    if extra:
        errors.append(f"{len(extra)} id(s) in your predictions are not in the gold split "
                      f"(e.g. {sorted(extra)[:3]}).")


def main():
    ap = argparse.ArgumentParser(description="Ayn-VQA submission format checker")
    ap.add_argument("--task", required=True, choices=["1a", "1b"])
    ap.add_argument("--pred", required=True, help="your prediction CSV")
    ap.add_argument("--gold", help="optional JSONL split to match ids against (e.g. ../data/task1a/devtest_en.jsonl)")
    args = ap.parse_args()

    print(f"Checking: {args.pred}  (task {args.task})\n")
    if not os.path.exists(args.pred):
        print(f"FORMAT CHECK FAILED — file not found: {args.pred}")
        sys.exit(1)

    rows, fieldnames = read_csv(args.pred)
    if not rows:
        print("FORMAT CHECK FAILED — file is empty or has no data rows.")
        sys.exit(1)

    gold_ids = load_gold_ids(args.gold) if args.gold else None
    if args.task == "1a":
        errors, warnings = check_1a(rows, fieldnames, gold_ids)
    else:
        errors, warnings = check_1b(rows, fieldnames, gold_ids)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
        print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  [x] {e}")
        print()
        print("FORMAT CHECK FAILED — fix the errors above before submitting.")
        sys.exit(1)

    print("FORMAT CHECK PASSED — your file is ready to zip and submit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
