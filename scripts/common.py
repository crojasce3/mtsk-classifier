"""Shared helpers for the analysis scripts. Run every script from the repository root."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
AREAS = ["T1", "T2", "T3", "T4", "T5"]
SEED = 7
N_BOOT = 2000


def load_labels():
    """Corpus labels (final, after curation), language, related-version group and CV fold."""
    lab = pd.read_csv(RES / "labels.csv").sort_values("row").reset_index(drop=True)
    return lab


def y_true():
    return np.array([AREAS.index(l) for l in load_labels()["label"]])


def load_probs(model):
    """Out-of-fold class probabilities (rows in corpus order, columns T1..T5)."""
    return np.load(RES / "cross_validation" / "oof_probabilities" / f"{model}.npy")


def load_folds():
    return json.load(open(RES / "cross_validation" / "folds.json"))


def macro_f1(a, b):
    return f1_score(a, b, labels=range(len(AREAS)), average="macro", zero_division=0)
