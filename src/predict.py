"""Predict intents for the test set with the final model.

Loads the fine-tuned model and the calibration (bias, transition prior, temperature) and writes
  test_predictions.csv          id,intent (the submission)
  test_predictions_review.csv   id,intent,confidence,needs_review

    python predict.py                                   # local model from train.py --full
    python predict.py --model henhua21/foursight-intent-e5   # published model, no training needed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibrate import decide
from data import LABELS, ROOT, load
from train import make_loader


@torch.no_grad()
def predict_logits(model, loader, device):
    # fp32 so CPU and GPU give the same labels
    model.eval()
    return torch.cat([model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu()
                      for batch in loader]).numpy()


def find_calibration(model):
    # local model: calibrate.py's output for that run; hub model: the file published with it
    if Path(model).exists():
        return ROOT / "outputs" / Path(model).name / "calibration.json"
    return hf_hub_download(model, "calibration.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "e5-ctx-drop50"))
    ap.add_argument("--calibration", default=None, help="defaults to the one that belongs to --model")
    ap.add_argument("--out", default=str(ROOT / "test_predictions.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.float32).to(device)
    calibration = args.calibration or find_calibration(args.model)
    params = {k: np.array(v) for k, v in json.load(open(calibration)).items()}

    test = load("test")  # sorted by id, with the previous two questions as context
    logits = predict_logits(model, make_loader(tokenizer, test, context=True), device)
    pred, conf = decide(logits, params)

    sub = pd.DataFrame({"id": test["id"], "intent": [LABELS[i] for i in pred]})
    sub.to_csv(args.out, index=False)
    review = sub.assign(confidence=conf.round(4), needs_review=conf < params["review_threshold"])
    review.to_csv(Path(args.out).with_name("test_predictions_review.csv"), index=False)

    # sanity checks against the submission rules
    ids = pd.read_csv(ROOT / "data" / "test.csv")["id"]
    assert len(sub) == len(ids) and set(sub["id"]) == set(ids), "ids don't match test.csv"
    assert sub["intent"].isin(LABELS).all(), "label outside the closed set"
    print(f"wrote {len(sub)} predictions to {args.out}")
    print(f"needs_review: {review['needs_review'].sum()} rows ({review['needs_review'].mean():.1%})")
    print(sub["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
