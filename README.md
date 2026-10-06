# FourSight intent classifier

Routes a FourSight chat question to one of seven intents. It uses the previous turns of the
conversation as context, and flags low-confidence predictions for human review.

- Model: `intfloat/multilingual-e5-base`, fine-tuned with the previous two questions as context
  ([Hugging Face](https://huggingface.co/henhua21/foursight-intent-e5),
  [W&B runs](https://wandb.ai/henhua21-tiktok/foursight-intent))
- Submission: `test_predictions.csv` (plus `test_predictions_review.csv` with confidence and
  `needs_review`)
- Deterministic: the same input always gives the same label (argmax, no sampling). Two runs
  give byte-identical files, and CPU and GPU give the same labels.

## Setup

Tested on Windows 11, Python 3.13, an RTX 4060 Ti (8 GB) and CUDA 12.8.

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Put the challenge files `train.csv` and `test.csv` in `data/`.

## Predict with the published model

No training needed. This downloads the model and its calibration from Hugging Face and
writes the same `test_predictions.csv`:

```bash
cd src
python predict.py --model henhua21/foursight-intent-e5
```

## Reproduce the submission from scratch

```bash
cd src
python train.py --context --name e5-ctx-drop50          # 5-fold CV, out-of-fold logits (~9 min)
python calibrate.py                                     # bias, transition prior, temperature (~1 min)
python train.py --context --name e5-ctx-drop50 --full   # final model on all rows -> models/ (~2 min)
python predict.py                                       # -> test_predictions.csv
```

Inference also runs on CPU. Training logs go to Weights & Biases in offline mode (`wandb/`);
run `wandb sync` to upload them.

## Experiments

```bash
python data.py         # data summary and fold check
python baselines.py    # TF-IDF and e5 baselines
python train.py        # fine-tuned e5 without context
python train.py --model microsoft/mdeberta-v3-base   # other backbones
python train.py --context --context_dropout 0 --name e5-ctx
```

Scores come from the official `score.py`. Full reports are written to `outputs/<model>/report.txt`.

## Results

5-fold grouped CV, out-of-fold predictions:

| model | OVERALL | macro-F1 | accuracy | fragment accuracy |
|---|---|---|---|---|
| TF-IDF + LR | 56.5 | 48.9 | 67.9 | 60.5 |
| multilingual-e5 + LR | 63.6 | 60.7 | 68.1 | 54.7 |
| multilingual-e5 + LR, context on fragments | 64.2 | 61.0 | 69.0 | 60.5 |
| fine-tuned mDeBERTa-v3-base | 56.3 | 50.5 | 64.9 | 58.9 |
| fine-tuned XLM-R-base | 56.9 | 51.2 | 65.5 | 58.3 |
| fine-tuned multilingual-e5-base | 57.7 | 51.6 | 67.0 | 58.3 |
| fine-tuned e5 + context (dropout 0.5) | 59.3 | 52.9 | 68.8 | 60.8 |

The fine-tuned models are trained with balanced class weights, so before any correction
they predict the rare classes far too often (chitchat recall 100%, precision 9%).
`calibrate.py` fixes this with a per-class bias. It also adds a transition prior (previous
predicted intent -> current intent) and calibrates the confidence. All of this is tuned inside
the CV, so the score below is not optimistic:

| final model | OVERALL | macro-F1 | accuracy | fragment accuracy | ECE |
|---|---|---|---|---|---|
| e5 + context, raw | 59.3 | 52.9 | 68.8 | 60.8 | 0.210 |
| + per-class bias + temperature | 66.9 | 63.1 | 72.7 | 61.8 | 0.041 |
| + transition prior | **67.7** | **63.8** | **73.5** | **66.0** | **0.035** |

`needs_review` (calibrated confidence < 0.5) flags 13% of questions. 60% of the flagged ones
are wrong, and accuracy on the rest is 78.5%.
