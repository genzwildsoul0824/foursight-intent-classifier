"""Baselines: TF-IDF + logistic regression and frozen multilingual-e5 embeddings + logistic regression.

All scores are out-of-fold on the grouped CV folds.
"""
import re

import numpy as np
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import LABELS, N_FOLDS, add_folds, load, score


def oof_proba(df, features, C):
    """Train one model per fold and predict the held-out fold."""
    proba = np.zeros((len(df), len(LABELS)))
    for k in range(N_FOLDS):
        is_train = (df["fold"] != k).to_numpy()
        X_train, X_val = features(is_train)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(X_train, df["label"][is_train])
        proba[~is_train] = clf.predict_proba(X_val)
    return proba


def to_labels(proba):
    return [LABELS[i] for i in proba.argmax(1)]


def replace_ids(text):
    # load numbers are unique, so as words they never repeat; one shared token is more useful
    return re.sub(r"\b(?=\w*\d)\w{5,}\b", " idtok ", text.lower())


def tfidf_features(df, is_train):
    # vectorizers are fit on the training folds only
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=replace_ids,
                           token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           preprocessor=replace_ids)
    word.fit(df["text"][is_train])
    char.fit(df["text"][is_train])
    X = hstack([word.transform(df["text"]), char.transform(df["text"])]).tocsr()
    return X[is_train], X[~is_train]


def e5_embed(model, texts):
    emb = model.encode(["query: " + t for t in texts], batch_size=32, normalize_embeddings=True)
    emb[np.array([t == "" for t in texts])] = 0  # no previous turn -> zero vector
    return emb


if __name__ == "__main__":
    df = add_folds(load("train"))

    # C values picked from a quick sweep over 1 / 4 / 16 / 64
    p_tfidf = oof_proba(df, lambda m: tfidf_features(df, m), C=4)
    score(df, to_labels(p_tfidf), "tfidf")

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    model.max_seq_length = 256
    cur = e5_embed(model, df["text"].tolist())
    ctx = np.hstack([cur] + [e5_embed(model, df[f"prev_{k}"].tolist()) for k in (1, 2)])

    p_e5 = oof_proba(df, lambda m: (cur[m], cur[~m]), C=16)
    p_ctx = oof_proba(df, lambda m: (ctx[m], ctx[~m]), C=16)
    score(df, to_labels(p_e5), "e5")
    score(df, to_labels(p_ctx), "e5+context")

    # context only for fragments: it helps short follow-ups but hurts full questions
    frag = df["is_fragment"].to_numpy()[:, None]
    score(df, to_labels(np.where(frag, p_ctx, p_e5)), "e5+context(frag)")
