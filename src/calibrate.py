"""Post-processing fitted on the out-of-fold logits: per-class bias, transition prior and
temperature. The reported score cross-validates this step too (fit on 4 folds, score the 5th).

    python calibrate.py --run e5-ctx-drop50
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from data import LABELS, N_FOLDS, ROOT, add_folds, load, score

ALPHAS = [0, 0.25, 0.5, 1.0]  # strengths tried for the transition prior
REVIEW_THRESHOLD = 0.5        # flag predictions with calibrated confidence below this


def overall(y, pred):
    return 0.6 * f1_score(y, pred, average="macro", labels=range(len(LABELS))) + 0.4 * (y == pred).mean()


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def tune_bias(logits, y):
    # coordinate search: one class at a time, a few rounds; on ties keep the smaller offset
    grid = sorted(np.round(np.arange(-3, 3.01, 0.1), 1), key=abs)
    bias = np.zeros(len(LABELS))
    for _ in range(3):
        for c in range(len(LABELS)):
            scores = []
            for v in grid:
                bias[c] = v
                scores.append(overall(y, (logits + bias).argmax(1)))
            bias[c] = grid[int(np.argmax(scores))]
    return bias


def fit_temperature(logits, y):
    # the temperature that minimises log loss; it doesn't change which class wins
    def nll(t):
        return -np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean()
    return min(np.arange(0.5, 5.01, 0.05), key=nll)


def transition_log_probs(y, rows):
    # P(intent of row i+1 | intent of row i), from consecutive rows that are both in `rows`
    counts = np.ones((len(LABELS), len(LABELS)))  # add-one smoothing
    pairs = rows[:-1] & rows[1:]
    np.add.at(counts, (y[:-1][pairs], y[1:][pairs]), 1)
    return np.log(counts / counts.sum(1, keepdims=True))


def prior(logits, p):
    # previous row's predicted intent -> log P(current intent | previous intent)
    prev = (logits + p["bias0"]).argmax(1)
    out = p["alpha"] * p["log_trans"][np.r_[0, prev[:-1]]]
    out[0] = 0  # first row has no previous turn
    return out


def fit(logits, y, rows, use_prior=True):
    """Tune bias, transition prior and temperature on the rows in `rows`."""
    p = {"bias0": tune_bias(logits[rows], y[rows]), "log_trans": transition_log_probs(y, rows)}
    best = None
    for alpha in (ALPHAS if use_prior else [0]):
        z = logits + prior(logits, {**p, "alpha": alpha})
        bias = tune_bias(z[rows], y[rows])
        s = overall(y[rows], (z[rows] + bias).argmax(1))
        if best is None or s > best[0]:
            best = (s, alpha, bias, z)
    _, p["alpha"], p["bias"], z = best
    p["temperature"] = fit_temperature(z[rows] + p["bias"], y[rows])
    return p


def decide(logits, p):
    probs = softmax((logits + prior(logits, p) + p["bias"]) / p["temperature"])
    return probs.argmax(1), probs.max(1)


def ece(conf, correct, bins=10):
    """Expected calibration error: average gap between confidence and accuracy."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def cross_validated(logits, y, fold, use_prior):
    pred = np.zeros(len(y), dtype=int)
    conf = np.zeros(len(y))
    for k in range(N_FOLDS):
        p = fit(logits, y, fold != k, use_prior)
        pk, ck = decide(logits, p)
        pred[fold == k], conf[fold == k] = pk[fold == k], ck[fold == k]
    return pred, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e5-ctx-drop50", help="folder in outputs/ with oof_logits.npy")
    args = ap.parse_args()
    out = ROOT / "outputs" / args.run

    df = add_folds(load("train"))
    y = df["label"].to_numpy()
    fold = df["fold"].to_numpy()
    logits = np.load(out / "oof_logits.npy")
    to_labels = lambda pred: [LABELS[i] for i in pred]

    print(f"raw:                 ECE {ece(softmax(logits).max(1), logits.argmax(1) == y):.3f}")
    score(df, to_labels(logits.argmax(1)), f"{args.run}-raw")
    pred, conf = cross_validated(logits, y, fold, use_prior=False)
    print(f"bias + temperature:  ECE {ece(conf, pred == y):.3f}")
    score(df, to_labels(pred), f"{args.run}-bias")
    pred, conf = cross_validated(logits, y, fold, use_prior=True)
    print(f"+ transition prior:  ECE {ece(conf, pred == y):.3f}")
    score(df, to_labels(pred), f"{args.run}-final")

    # precision = flagged rows that are wrong, recall = wrong rows that get flagged
    wrong = pred != y
    print("\nthreshold  flagged  precision  recall  accuracy on the rest")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        flag = conf < t
        print(f"   {t:.1f}     {flag.mean():5.1%}    {wrong[flag].mean():5.1%}    {wrong[flag].sum() / wrong.sum():5.1%}"
              f"      {1 - wrong[~flag].mean():5.1%}")

    # final parameters for the test set, tuned on all out-of-fold predictions
    p = fit(logits, y, np.ones(len(y), dtype=bool))
    p["review_threshold"] = REVIEW_THRESHOLD
    p = {k: np.asarray(v).round(3).tolist() for k, v in p.items()}
    (out / "calibration.json").write_text(json.dumps(p, indent=1))
    print(f"\nalpha {p['alpha']}, temperature {p['temperature']}, bias {dict(zip(LABELS, p['bias']))}")


if __name__ == "__main__":
    main()
