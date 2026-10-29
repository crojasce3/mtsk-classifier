"""
Cross-validation results reported in Results (Classification performance) and
the nested review threshold reported in Results (Qualitative error analysis).

Inputs : results/labels.csv, results/cross_validation/{folds.json, oof_probabilities/*.npy}
Outputs: printed tables and figures/e5-large_confusion_matrix_cv.png

Usage  : python scripts/cv_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)

from common import AREAS, FIG, N_BOOT, SEED, load_folds, load_labels, load_probs, macro_f1, y_true

MODELS = ["e5-large", "tfidf-lr", "minilm", "xlmr-base", "xlmr-base-2e5"]

y = y_true()
lab = load_labels()
folds = load_folds()
N = len(y)

# Bootstrap indices shared by all models (paired comparisons use the same resamples)
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, N, N) for _ in range(N_BOOT)]

print("=" * 90, "\nTable 3. Cross-validated performance\n" + "=" * 90)
rows, preds = [], {}
for m in MODELS:
    p = load_probs(m).argmax(1)
    preds[m] = p
    f1s = [macro_f1(y[f["test"]], p[f["test"]]) for f in folds]
    mean, sd = np.mean(f1s), np.std(f1s, ddof=1)
    h = stats.t.ppf(0.975, len(f1s) - 1) * sd / np.sqrt(len(f1s))
    bs = [macro_f1(y[i], p[i]) for i in boot]
    rows.append({"model": m, "folds": " ".join(f"{v:.3f}" for v in f1s),
                 "macroF1_mean": mean, "sd": sd, "t_ci_low": mean - h, "t_ci_high": mean + h,
                 "pooled_macroF1": macro_f1(y, p),
                 "boot_ci_low": np.percentile(bs, 2.5), "boot_ci_high": np.percentile(bs, 97.5),
                 "accuracy": accuracy_score(y, p)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nPaired bootstrap, macro-F1 difference e5-large minus each alternative")
pe = preds["e5-large"]
for m in MODELS[1:]:
    d = [macro_f1(y[i], pe[i]) - macro_f1(y[i], preds[m][i]) for i in boot]
    print(f"  {m:14s} diff = {macro_f1(y, pe) - macro_f1(y, preds[m]):.3f}  "
          f"95% CI [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]")

print("\nTable 4. Per-area performance of multilingual-e5-large (pooled out-of-fold)")
print(classification_report(y, pe, target_names=AREAS, digits=3, zero_division=0))
cm = confusion_matrix(y, pe)
print(pd.DataFrame(cm, index=AREAS, columns=AREAS))

print("\nBy language (e5-large)")
for lg in ["es", "en", "pt"]:
    msk = (lab["language"] == lg).values
    print(f"  {lg}: n={msk.sum():3d}  accuracy={accuracy_score(y[msk], pe[msk]):.3f}  "
          f"macro-F1={macro_f1(y[msk], pe[msk]):.3f}")

# Figure 1: row-normalised confusion matrix
FIG.mkdir(exist_ok=True)
cmn = cm / cm.sum(1, keepdims=True)
annot = np.array([[f"{cmn[i, j]:.2f}\n({cm[i, j]})" for j in range(5)] for i in range(5)])
plt.figure(figsize=(5.2, 4.3))
sns.heatmap(cmn, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, xticklabels=AREAS,
            yticklabels=AREAS, cbar_kws={"label": "Proportion of true-class documents"})
plt.xlabel("Predicted area"); plt.ylabel("True area"); plt.tight_layout()
plt.savefig(FIG / "e5-large_confusion_matrix_cv.png", dpi=300)

print("\n" + "=" * 90 + "\n128 versus 256 tokens (multilingual-e5-large)\n" + "=" * 90)
p256 = load_probs("e5-large-256").argmax(1)
f128 = np.array([macro_f1(y[f["test"]], pe[f["test"]]) for f in folds])
f256 = np.array([macro_f1(y[f["test"]], p256[f["test"]]) for f in folds])
print("fold macro-F1 128:", f128.round(4), "| 256:", f256.round(4))
print(f"mean 256 = {f256.mean():.3f} +/- {f256.std(ddof=1):.3f}; pooled 256 = {macro_f1(y, p256):.3f}")
t = stats.ttest_rel(f256, f128)
print(f"paired t-test across folds: t(4) = {t.statistic:.2f}, p = {t.pvalue:.2f}")
rng2 = np.random.default_rng(SEED)
d = []
for _ in range(N_BOOT):
    i = rng2.integers(0, N, N)
    d.append(macro_f1(y[i], p256[i]) - macro_f1(y[i], pe[i]))
print(f"paired bootstrap diff = {macro_f1(y, p256) - macro_f1(y, pe):.3f} "
      f"95% CI [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]")
b = int(((pe != y) & (p256 == y)).sum()); c = int(((pe == y) & (p256 != y)).sum())
print(f"256 corrects {b} errors, introduces {c}; McNemar exact p = {stats.binomtest(b, b + c, 0.5).pvalue:.2f}")
f1a = classification_report(y, pe, output_dict=True, zero_division=0)
f1b = classification_report(y, p256, output_dict=True, zero_division=0)
print("per-area F1 128 -> 256:", {AREAS[k]: (round(f1a[str(k)]['f1-score'], 3), round(f1b[str(k)]['f1-score'], 3)) for k in range(5)})
c128, c256 = confusion_matrix(y, pe), confusion_matrix(y, p256)
print(f"T3<->T4 confusions: {c128[2,3] + c128[3,2]} -> {c256[2,3] + c256[3,2]}; "
      f"T5->T4: {c128[4,3]} -> {c256[4,3]}")

print("\n" + "=" * 90 + "\nConfidence as review criterion (Qualitative error analysis)\n" + "=" * 90)
P = load_probs("e5-large"); conf = P.max(1); err = pe != y
fold_of = lab["cv_fold"].values
print(f"errors: {err.sum()} | errors with confidence > 0.95: {(err & (conf > 0.95)).sum()} | "
      f"AUROC = {roc_auc_score(err, -conf):.3f}")
grid = np.round(np.arange(0.40, 0.996, 0.005), 3)
review = np.zeros(N, bool); chosen = []
for k in range(5):
    tr, te = fold_of != k, fold_of == k
    thr = min(t_ for t_ in grid if err[tr & (conf < t_)].sum() / err[tr].sum() >= 2 / 3)
    chosen.append(thr); review[te] = conf[te] < thr
print(f"nested thresholds per fold: {chosen}")
print(f"review {100 * review.mean():.1f}% of documents, detect {100 * err[review].sum() / err.sum():.1f}% "
      f"of errors, accuracy of accepted predictions = {1 - err[~review].mean():.3f}")
