"""
Ayn-VQA Official Scorer (Task 1a / Task 1b)
===========================================
Computes exactly the same metrics as the Codabench leaderboard, so you can
score yourself locally on a labelled split (e.g. ``dev``) before submitting.

Gold labels are read straight from the released HuggingFace JSONL:
    * Task 1a — the ``label`` field (index 0/1/2 of the correct option).
    * Task 1b — the ``labels`` field (list of three booleans, exactly one True).
Predictions are the submission CSVs (see the format checker / README).

Usage
-----
    # Task 1a — predict the option index
    python score.py --task 1a --gold ../data/task1a/dev_en.jsonl --pred prediction.csv

    # Task 1b — True/False per statement
    python score.py --task 1b --gold ../data/task1b/dev_en.jsonl --pred prediction.csv

A missing or unparseable prediction always counts as wrong. The first reported
metric is the ranking metric used for the leaderboard.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbone import evaluate_tf  # bundled alongside this scorer (same as Codabench)


# ────────────────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────────────────
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _require(cond, msg):
    if not cond:
        print(f"ERROR: {msg}")
        sys.exit(1)


def _rate(n, d):
    return float(round(n / d, 6)) if d else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Task 1a — Spoken VQA (accuracy + balanced accuracy + macro-F1)
# ────────────────────────────────────────────────────────────────────────────
def score_1a(gold_rows, pred_rows):
    _require(gold_rows and "label" in gold_rows[0],
             "gold JSONL has no 'label' field — use a labelled split (train/dev).")
    _require(pred_rows and {"id", "prediction"}.issubset(pred_rows[0].keys()),
             "prediction CSV must have columns: id,prediction")

    truth = {str(r["id"]).strip(): str(r["label"]).strip() for r in gold_rows}
    pred = {}
    for r in pred_rows:
        iid = str(r.get("id", "")).strip()
        _require(iid not in pred, f"duplicate id in prediction: {iid}")
        pred[iid] = str(r.get("prediction", "")).strip()

    covered = sum(1 for i in truth if i in pred)
    if covered < len(truth):
        print(f"WARNING: predictions cover {covered}/{len(truth)} ids (missing -> wrong)")

    classes = sorted(set(truth.values()))
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    correct = 0
    for i, t in truth.items():
        p = pred.get(i)
        if p == t:
            correct += 1
            tp[t] += 1
        else:
            fn[t] += 1
            if p in tp:
                fp[p] += 1
    accuracy = correct / len(truth) if truth else 0.0

    recalls, f1s = [], []
    for c in classes:
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        f1s.append(f1)
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return [
        ("accuracy",          round(accuracy, 6),          True),
        ("balanced_accuracy", round(balanced_accuracy, 6), False),
        ("macro_f1",          round(macro_f1, 6),          False),
    ]


# ────────────────────────────────────────────────────────────────────────────
# Task 1b — Hallucination detection (full evals.py panel)
# ────────────────────────────────────────────────────────────────────────────
def score_1b(gold_rows, pred_rows):
    _require(gold_rows and "labels" in gold_rows[0],
             "gold JSONL has no 'labels' field — use a labelled split (train/dev).")
    _require(pred_rows and {"id", "statement_index", "prediction"}.issubset(pred_rows[0].keys()),
             "prediction CSV must have columns: id,statement_index,prediction")

    # gold: grounded (True) statement index per item
    truth = {}
    for r in gold_rows:
        labels = r["labels"]
        truth[str(r["id"]).strip()] = labels.index(True)

    preds = {}  # id -> {statement_index -> raw prediction text}
    for r in pred_rows:
        iid = str(r["id"]).strip()
        try:
            si = int(str(r["statement_index"]).strip())
        except (ValueError, TypeError):
            continue
        preds.setdefault(iid, {})[si] = r["prediction"]

    total = q_minus_total = covered = 0
    q_plus_c = q_minus_c = combined_c = 0
    cfhr_2 = cfhr_2_total = cfhr_3 = cfhr_3_total = 0
    qp_no_clear = qm0_no_clear = qm1_no_clear = 0

    for iid, true_idx in truth.items():
        total += 1
        q_minus_total += 2
        pr = preds.get(iid)
        if pr:
            covered += 1
        pr = pr or {}
        ev = {i: evaluate_tf(pr.get(i, "")) for i in range(3)}
        false_idx = [i for i in range(3) if i != true_idx]
        tq, fq0, fq1 = ev[true_idx], ev[false_idx[0]], ev[false_idx[1]]

        inc_qp = (tq.pred == "true")
        inc_qm0 = (fq0.pred == "false")
        inc_qm1 = (fq1.pred == "false")

        q_plus_c += inc_qp
        q_minus_c += inc_qm0 + inc_qm1
        combined_c += inc_qp and inc_qm0 and inc_qm1

        if inc_qp:
            cfhr_2_total += 1
            if not (inc_qm0 and inc_qm1):
                cfhr_2 += 1
        if inc_qp or inc_qm0 or inc_qm1:
            cfhr_3_total += 1
            if not (inc_qp and inc_qm0 and inc_qm1):
                cfhr_3 += 1

        qp_no_clear += (tq.pred is None)
        qm0_no_clear += (fq0.pred is None)
        qm1_no_clear += (fq1.pred is None)

    if covered < total:
        print(f"WARNING: predictions cover {covered}/{total} ids (missing -> wrong)")

    q_plus_accuracy = _rate(q_plus_c, total)
    combined_accuracy = _rate(combined_c, total)

    return [
        ("contrastive_instability", _rate(cfhr_3, cfhr_3_total),     True),   # ranking; lower is better
        ("combined_accuracy",       combined_accuracy,               False),
        ("cfhr",                    _rate(cfhr_2, cfhr_2_total),     False),
        ("q_plus_accuracy",         q_plus_accuracy,                 False),
        ("q_minus_accuracy",        _rate(q_minus_c, q_minus_total), False),
    ]


def main():
    ap = argparse.ArgumentParser(description="Ayn-VQA official scorer (Task 1a / 1b)")
    ap.add_argument("--task", required=True, choices=["1a", "1b"])
    ap.add_argument("--gold", required=True, help="labelled JSONL (e.g. ../data/task1a/dev_en.jsonl)")
    ap.add_argument("--pred", required=True, help="your prediction CSV")
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_csv(args.pred)
    scores = score_1a(gold_rows, pred_rows) if args.task == "1a" else score_1b(gold_rows, pred_rows)

    print(f"\nScored Task {args.task} on {len(gold_rows)} items\n")
    print(f"{'metric':<34}{'value':>10}")
    print("-" * 44)
    for name, value, primary in scores:
        marker = "  <-- RANKING" if primary else ""
        print(f"  {name:<32}{value:>10.4f}{marker}")
    print("-" * 44)


if __name__ == "__main__":
    main()
