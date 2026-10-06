"""Fine-tune a multilingual encoder with 5-fold grouped CV.

For every fold we train on the other 4 folds, then save the logits on the held-out fold
(out-of-fold) and on the test set. No checkpoints are kept during CV.

    python train.py --model intfloat/multilingual-e5-base
"""
import argparse
import gc
import os

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def class_weights(labels):
    # same as sklearn's class_weight="balanced": rare classes count as much as common ones
    counts = np.bincount(labels, minlength=len(LABELS))
    return torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float)


def make_loader(tokenizer, texts, labels=None, shuffle=False, seed=0, batch_size=16):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(texts))]
    if labels is not None:
        for item, y in zip(items, labels):
            item["labels"] = int(y)

    def collate(batch):
        out = tokenizer.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch], return_tensors="pt")
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
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

    tr, va = df[df["fold"] != k], df[df["fold"] == k]
    train_loader = make_loader(tokenizer, tr["text"], tr["label"], shuffle=True, seed=args.seed + k, batch_size=args.batch_size)
    val_loader = make_loader(tokenizer, va["text"], va["label"])

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

    test_logits = predict(model, make_loader(tokenizer, test["text"]))
    return val_logits, test_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
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

    oof = np.zeros((len(df), len(LABELS)), dtype=np.float32)
    test_logits = []
    for k in range(N_FOLDS):
        val_logits, t_logits = train_fold(args, df, test, k)
        # free the last fold's model before loading the next one (8 GB card)
        gc.collect()
        torch.cuda.empty_cache()
        oof[(df["fold"] == k).to_numpy()] = val_logits
        test_logits.append(t_logits)

    np.save(out / "oof_logits.npy", oof)
    np.save(out / "test_logits.npy", np.stack(test_logits))

    res = score(df, [LABELS[i] for i in oof.argmax(1)], name)
    wandb.summary.update({"OVERALL": res["OVERALL"], "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
                          "fragment_acc": res["fragment_acc"],
                          **{f"f1/{c}": res["per_class"][c]["f1"] for c in LABELS}})
    wandb.finish()


if __name__ == "__main__":
    main()
