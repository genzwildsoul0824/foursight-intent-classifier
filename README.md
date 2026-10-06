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
python data.py    # data summary and fold check
```
