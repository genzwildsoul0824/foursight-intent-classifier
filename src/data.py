"""Loading the data, adding conversation context and building the CV folds."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
LABELS = json.load(open(ROOT / "labels.json"))["labels"]

SEED = 42
N_FOLDS = 5
BLOCK_SIZE = 20       # ids follow conversation order, so 20 consecutive rows ~ one session
N_PREV = 2            # previous turns used as context
PREV_MAX_CHARS = 200  # previous turns can be pasted tables, keep only the start


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def is_fragment(text):
    # short follow-ups ("yes", "both", "october 1") or bare load ids
    tokens = text.split()
    return len(tokens) <= 3 or all(any(c.isdigit() for c in t) for t in tokens)


def load(split):
    df = pd.read_csv(ROOT / "data" / f"{split}.csv", keep_default_na=False)
    df = df.sort_values("id").reset_index(drop=True)
    df["text"] = df["question"].map(clean)
    df["is_fragment"] = df["text"].map(is_fragment)
    # context = the previous questions by id (text only, never labels, so it works on test too)
    for k in range(1, N_PREV + 1):
        df[f"prev_{k}"] = df["text"].shift(k, fill_value="").str[:PREV_MAX_CHARS]
    if "intent" in df:
        df["intent"] = df["intent"].str.strip().str.lower()
        df["label"] = df["intent"].map(LABELS.index)
    return df


def add_folds(df):
    # group = block of consecutive ids; duplicates join the group of their first copy so the
    # same text never ends up in two folds
    groups = ((df["id"] - 1) // BLOCK_SIZE).to_numpy().copy()
    first_group = {}
    for i, key in enumerate(df["text"].str.lower()):
        groups[i] = first_group.setdefault(key, groups[i])

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, val_idx) in enumerate(cv.split(df, df["label"], groups)):
        df.loc[val_idx, "fold"] = k
    return df


def score(df, pred, name):
    """Score predictions with the official score.py and print a one-line summary."""
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    df[["id", "question", "intent"]].to_csv(out / "gold.csv", index=False)
    pd.DataFrame({"id": df["id"], "intent": pred}).to_csv(out / "pred.csv", index=False)
    with open(out / "report.txt", "w") as f:
        subprocess.run([sys.executable, ROOT / "score.py", "--gold", out / "gold.csv",
                        "--pred", out / "pred.csv", "--out", out / "score.json"], stdout=f, check=True)
    res = json.load(open(out / "score.json"))

    correct = np.asarray(pred) == df["intent"].to_numpy()
    res["fragment_acc"] = correct[df["is_fragment"].to_numpy()].mean()
    print(f"{name:<16} OVERALL {100 * res['OVERALL']:.1f}   macro-F1 {100 * res['macro_f1']:.1f}   "
          f"acc {100 * res['accuracy']:.1f}   fragment acc {100 * res['fragment_acc']:.1f}")
    return res


if __name__ == "__main__":
    train, test = load("train"), load("test")
    train = add_folds(train)
    print(f"train {len(train)}, test {len(test)}")
    print(f"fragments: train {train['is_fragment'].sum()}, test {test['is_fragment'].sum()}\n")
    print(pd.crosstab(train["intent"], train["fold"], margins=True))

    folds_per_text = train.groupby(train["text"].str.lower())["fold"].nunique()
    print(f"\nduplicate texts split across folds: {(folds_per_text > 1).sum()}")
    print(f"neighbouring rows in the same fold: {(train['fold'].diff() == 0).mean():.0%}")
