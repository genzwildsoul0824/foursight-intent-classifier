"""Fine-tune a multilingual encoder with 5-fold grouped CV.

For every fold we train on the other 4 folds, then save the logits on the held-out fold
(out-of-fold) and on the test set. No checkpoints are kept during CV.

    python train.py                                   # current question only
    python train.py --context --name e5-ctx-drop50    # + previous turns, context dropout 0.5
"""
import argparse
import gc
import os
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for deterministic cuBLAS
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import LABELS, N_FOLDS, ROOT, add_folds, load, score

MAX_LEN = 128  # median question is 14 tokens, only 36 of 1904 are longer than 128


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def class_weights(labels):
    # same as sklearn's class_weight="balanced": rare classes count as much as common ones
    counts = np.bincount(labels, minlength=len(LABELS))
    return torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float)


def encode(tokenizer, df, context):
    if not context:
        return tokenizer(list(df["text"]), truncation=True, max_length=MAX_LEN)
    # (question, previous turns) as a text pair; longest_first trims the longer part first
    prev = [" | ".join(t for t in turns if t) for turns in zip(df["prev_1"], df["prev_2"])]
    return tokenizer(list(df["text"]), prev, truncation="longest_first", max_length=MAX_LEN)


def make_loader(tokenizer, df, context=False, p_drop=0.0, labels=False, shuffle=False, seed=0, batch_size=16):
    plain = encode(tokenizer, df, context=False)
    with_ctx = encode(tokenizer, df, context=True) if context else None

    def collate(rows):
        batch = []
        for i in rows:
            # context dropout: sometimes train on the question alone
            enc = with_ctx if with_ctx is not None and random.random() >= p_drop else plain
            batch.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]})
        out = tokenizer.pad(batch, return_tensors="pt")
        if labels:
            out["labels"] = torch.tensor(df["label"].to_numpy()[rows])
        return out

    return DataLoader(range(len(df)), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits = []
    for batch in loader:
        batch = {key: t.cuda() for key, t in batch.items() if key != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(**batch).logits.float().cpu())
    return torch.cat(logits).numpy()


def train_fold(args, df, test, k):
    set_seed(args.seed + k)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # force fp32 weights: mdeberta's checkpoint is fp16 and trains to NaN otherwise
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), dtype=torch.float32).cuda()
    # freeze the 250k-token embedding table (~70% of the weights), saves ~2 GB of GPU memory
    model.base_model.embeddings.word_embeddings.weight.requires_grad = False

    tr = df[df["fold"] != k].reset_index(drop=True)
    va = df[df["fold"] == k].reset_index(drop=True)
    train_loader = make_loader(tokenizer, tr, args.context, args.context_dropout, labels=True,
                               shuffle=True, seed=args.seed + k, batch_size=args.batch_size)
    val_loader = make_loader(tokenizer, va, args.context)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(tr["label"]).cuda(), label_smoothing=0.1)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: t.cuda() for key, t in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(**batch).logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            wandb.log({f"fold{k}/train_loss": loss.item()})

        val_logits = predict(model, val_loader)
        val_loss = loss_fn(torch.tensor(val_logits).cuda(), torch.tensor(va["label"].to_numpy()).cuda()).item()
        val_acc = (val_logits.argmax(1) == va["label"].to_numpy()).mean()
        wandb.log({f"fold{k}/val_loss": val_loss, f"fold{k}/val_acc": val_acc, f"fold{k}/epoch": epoch + 1})
        print(f"fold {k} epoch {epoch + 1}: val loss {val_loss:.3f}, val acc {val_acc:.3f}")

    # with context, also predict without it for comparison
    variants = [True, False] if args.context else [False]
    val = {c: predict(model, make_loader(tokenizer, va, c)) for c in variants}
    tst = {c: predict(model, make_loader(tokenizer, test, c)) for c in variants}
    return val, tst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--context", action="store_true", help="add the previous two questions as context")
    ap.add_argument("--context_dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or args.model.split("/")[-1]

    df = add_folds(load("train"))
    test = load("test")
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    wandb.init(project="foursight-intent", name=name, config=vars(args), dir=ROOT)

    variants = [True, False] if args.context else [False]
    oof = {c: np.zeros((len(df), len(LABELS)), dtype=np.float32) for c in variants}
    test_logits = {c: [] for c in variants}
    for k in range(N_FOLDS):
        val, tst = train_fold(args, df, test, k)
        # free the last fold's model before loading the next one (8 GB card)
        gc.collect()
        torch.cuda.empty_cache()
        for c in variants:
            oof[c][(df["fold"] == k).to_numpy()] = val[c]
            test_logits[c].append(tst[c])

    for c in variants:
        suffix = "" if c == args.context else "_noctx"
        np.save(out / f"oof_logits{suffix}.npy", oof[c])
        np.save(out / f"test_logits{suffix}.npy", np.stack(test_logits[c]))

    to_labels = lambda logits: [LABELS[i] for i in logits.argmax(1)]
    res = score(df, to_labels(oof[args.context]), name)
    if args.context:
        score(df, to_labels(oof[False]), f"{name}-noctx")
        frag = df["is_fragment"].to_numpy()[:, None]
        score(df, to_labels(np.where(frag, oof[True], oof[False])), f"{name}-frag")

    wandb.summary.update({"OVERALL": res["OVERALL"], "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
                          "fragment_acc": res["fragment_acc"],
                          **{f"f1/{c}": res["per_class"][c]["f1"] for c in LABELS}})
    wandb.finish()


if __name__ == "__main__":
    main()
