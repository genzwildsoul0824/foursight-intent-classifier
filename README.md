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
```

Scores come from the official `score.py`. Full reports are written to `outputs/<model>/report.txt`.

## Results so far

5-fold grouped CV, out-of-fold predictions:

| model | OVERALL | macro-F1 | accuracy | fragment accuracy |
|---|---|---|---|---|
| TF-IDF + LR | 56.5 | 48.9 | 67.9 | 60.5 |
| multilingual-e5 + LR | 63.6 | 60.7 | 68.1 | 54.7 |
| multilingual-e5 + LR, context on fragments | 64.2 | 61.0 | 69.0 | 60.5 |
