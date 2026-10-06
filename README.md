# FourSight intent classifier

Routes a FourSight chat question to one of seven intents. Uses the previous turns of the
conversation as context. Work in progress.

## Setup

Tested on Windows 11, Python 3.13, an RTX 4060 Ti (8 GB) and CUDA 12.8.

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Put the challenge files `train.csv` and `test.csv` in `data/`.

## Run

```bash
cd src
python data.py         # data summary and fold check
python baselines.py    # TF-IDF and e5 baselines
python train.py        # fine-tune multilingual-e5 (5 folds, ~7 min on an RTX 4060 Ti)
python train.py --model microsoft/mdeberta-v3-base   # other backbones
python train.py --context --name e5-ctx-drop50       # + previous two turns, context dropout 0.5
python predict.py                                    # bias, transition prior, calibration, needs_review
```

Training logs to Weights & Biases in offline mode (`wandb/`); run `wandb sync` to upload.

Scores come from the official `score.py`. Full reports are written to `outputs/<model>/report.txt`.

## Results so far

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
`predict.py` fixes this with a per-class bias, adds a transition prior (previous predicted
intent -> current intent) and calibrates the confidence. All of this is tuned inside the CV,
so the score below is not optimistic:

| final model | OVERALL | macro-F1 | accuracy | fragment accuracy | ECE |
|---|---|---|---|---|---|
| e5 + context, raw | 59.3 | 52.9 | 68.8 | 60.8 | 0.210 |
| + per-class bias + temperature | 66.9 | 63.1 | 72.7 | 61.8 | 0.041 |
| + transition prior | **67.7** | **63.8** | **73.5** | **66.0** | **0.035** |

`needs_review` (calibrated confidence < 0.5) flags 13% of questions. 60% of the flagged ones
are wrong, and accuracy on the rest is 78.5%.
