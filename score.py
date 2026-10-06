#!/usr/bin/env python3
"""Official scorer for the FourSight Intent Classification challenge.

    python3 score.py --gold data/test.csv --pred your_predictions.csv

Both files are CSV. The gold file has columns `id,question,intent` (the scorer only reads
`id` and `intent`). The prediction file has columns `id,intent` (see sample_submission.csv).
A row missing from the prediction file, or predicted with a label outside the closed set,
scores as wrong. Standard library only.

What is scored
  accuracy     fraction of rows whose predicted intent equals the gold intent.
  per class    precision, recall and F1 for each of the seven labels.
  macro-F1     unweighted mean of the per-class F1s (every class counts equally, so the
               rare classes are not drowned out by the common ones).
  weighted-F1  per-class F1 weighted by support (reported for reference).

  OVERALL = 0.6 * macro-F1 + 0.4 * accuracy

We also print the confusion matrix and the most-confused class pairs.

Normalisation: labels are lower-cased and stripped before comparison; nothing else.
"""
import argparse
import csv
import sys
from collections import defaultdict

LABELS = [
    "shipment_information",
    "realtime_query",
    "analytics",
    "disruptions",
    "chitchat",
    "knowledge_base",
    "other",
]
LABEL_SET = set(LABELS)


def norm(v):
    return (v or "").strip().lower()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(rows, path):
    out = {}
    for r in rows:
        if "id" not in r or "intent" not in r:
            sys.exit(f"{path}: expected columns including 'id' and 'intent', got {list(r)}")
        out[str(r["id"]).strip()] = norm(r["intent"])
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", required=True, help="gold CSV with columns id,question,intent")
    ap.add_argument("--pred", required=True, help="prediction CSV with columns id,intent")
    ap.add_argument("--out", help="optional: write the full report here as JSON")
    a = ap.parse_args()

    gold = load_labels(read_csv(a.gold), a.gold)
    pred = load_labels(read_csv(a.pred), a.pred)
    if not gold:
        sys.exit(f"no gold rows found in {a.gold}")

    missing = [d for d in gold if d not in pred]
    unknown_ids = [d for d in pred if d not in gold]
    bad_label = sorted({v for d, v in pred.items() if d in gold and v not in LABEL_SET})

    # Confusion matrix: conf[gold][pred]. A missing prediction becomes "__missing__".
    labels_ext = LABELS + ["__missing__"]
    conf = {g: defaultdict(int) for g in LABELS}
    correct = 0
    for d, g in gold.items():
        p = pred.get(d, "__missing__")
        if p not in LABEL_SET and p != "__missing__":
            p = "__invalid__"
            conf[g].setdefault("__invalid__", 0)
        conf[g][p] += 1
        if p == g:
            correct += 1

    n = len(gold)
    accuracy = correct / n

    # Per-class P/R/F1 (a prediction counted only when it is a valid label).
    per = {}
    for c in LABELS:
        tp = conf[c].get(c, 0)
        fn = sum(v for k, v in conf[c].items() if k != c)
        fp = sum(conf[g].get(c, 0) for g in LABELS if g != c)
        p, r, f = prf(tp, fp, fn)
        per[c] = {"precision": p, "recall": r, "f1": f, "support": sum(conf[c].values())}

    macro_f1 = sum(per[c]["f1"] for c in LABELS) / len(LABELS)
    weighted_f1 = sum(per[c]["f1"] * per[c]["support"] for c in LABELS) / n
    overall = 0.6 * macro_f1 + 0.4 * accuracy

    # Most-confused ordered pairs (gold -> pred, pred != gold).
    pairs = []
    for g in LABELS:
        for p, cnt in conf[g].items():
            if p != g and cnt > 0:
                pairs.append((cnt, g, p))
    pairs.sort(reverse=True)

    print(f"rows scored          {n}   (missing from submission: {len(missing)}, "
          f"unknown ids ignored: {len(unknown_ids)})")
    if bad_label:
        print(f"WARNING invalid labels in submission (scored wrong): {bad_label}")
    print(f"accuracy             {100 * accuracy:6.2f}%")
    print(f"macro-F1             {100 * macro_f1:6.2f}%")
    print(f"weighted-F1          {100 * weighted_f1:6.2f}%")
    print(f"OVERALL              {100 * overall:6.2f}")
    print()
    print(f"{'class':<22} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}")
    for c in LABELS:
        m = per[c]
        print(f"{c:<22} {100*m['precision']:6.1f}% {100*m['recall']:6.1f}% "
              f"{100*m['f1']:6.1f}% {m['support']:8d}")
    print()
    print("most confused (gold -> predicted):")
    for cnt, g, p in pairs[:8]:
        lbl = p if p not in ("__missing__", "__invalid__") else p.strip("_")
        print(f"   {g:<22} -> {lbl:<22} {cnt}")
    print()
    print("confusion matrix (rows = gold, cols = predicted):")
    cols = labels_ext
    head = " " * 22 + "".join(f"{c[:8]:>9}" for c in cols)
    print(head)
    for g in LABELS:
        row = "".join(f"{conf[g].get(c, 0):>9}" for c in cols)
        print(f"{g:<22}{row}")

    if a.out:
        import json
        report = {
            "n": n, "accuracy": accuracy, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "OVERALL": overall,
            "per_class": per,
            "confusion": {g: dict(conf[g]) for g in LABELS},
            "missing": len(missing), "invalid_labels": bad_label,
        }
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
