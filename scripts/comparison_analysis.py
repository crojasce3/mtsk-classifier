"""
Comparison of the fine-tuned classifier with the prompted LLM and the independent expert
(Results: comparison with expert judgement and with a large language model;
Tables 5 and 6, Fig 4).

Inputs : results/labels.csv, results/cross_validation/oof_probabilities/e5-large.npy,
         results/llm/deepseek_predictions.csv, results/expert/expert_annotation.csv
Outputs: printed results and figures/expert_vs_model_confusion.png

Usage  : python scripts/comparison_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, fisher_exact
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

from common import AREAS, FIG, N_BOOT, RES, SEED, load_labels, load_probs, macro_f1, y_true

y = y_true(); lab = load_labels(); N = len(y)
P = load_probs("e5-large"); e5 = P.argmax(1); e5_conf = P.max(1)
llm_df = pd.read_csv(RES / "llm" / "deepseek_predictions.csv").sort_values("row").reset_index(drop=True)
llm = np.array([AREAS.index(l) for l in llm_df["llm_label"]])
ex = pd.read_csv(RES / "expert" / "expert_annotation.csv")
idx = ex["row"].values
hum = np.array([AREAS.index(l) for l in ex["expert_label"]])
kappa = lambda a, b: cohen_kappa_score(a, b)

print("=" * 90 + "\nPrompted LLM on the whole corpus (Table 5)\n" + "=" * 90)
print(f"API model: {llm_df['api_model'].unique()} | dates: {llm_df['query_date_utc'].min()} - {llm_df['query_date_utc'].max()}")
print(f"accuracy={accuracy_score(y, llm):.3f} macro-F1={macro_f1(y, llm):.3f} kappa={kappa(y, llm):.3f}")
print(classification_report(y, llm, target_names=AREAS, digits=3, zero_division=0))
print(pd.DataFrame(confusion_matrix(y, llm), index=AREAS, columns=AREAS))
print("predicted per area:", dict(zip(AREAS, np.bincount(llm, minlength=5))))
for lg in ["es", "en", "pt"]:
    m = (lab["language"] == lg).values
    print(f"  {lg}: accuracy={accuracy_score(y[m], llm[m]):.3f}")

rng = np.random.default_rng(SEED); d = []
for _ in range(N_BOOT):
    i = rng.integers(0, N, N); d.append(macro_f1(y[i], llm[i]) - macro_f1(y[i], e5[i]))
print(f"\nLLM minus e5-large macro-F1 = {macro_f1(y, llm) - macro_f1(y, e5):.3f} "
      f"95% CI [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]")
b = int(((llm == y) & (e5 != y)).sum()); c = int(((llm != y) & (e5 == y)).sum())
print(f"LLM correct where e5 wrong: {b}; e5 correct where LLM wrong: {c}; McNemar p = {binomtest(b, b + c, 0.5).pvalue:.3f}")
le, ee = llm != y, e5 != y
print(f"shared errors: {(le & ee).sum()} of {le.sum()} LLM errors; same wrong area in {((llm == e5) & le & ee).sum()}")
sec = llm_df["llm_secondary"].fillna("").astype(str)
print("LLM errors with the corpus label among its secondary areas:",
      sum(AREAS[t] in s for t, s, p in zip(y, sec, llm) if p != t), "of", le.sum())
agree = llm == e5
print(f"models agree on {agree.sum()} documents (accuracy {(e5[agree] == y[agree]).mean():.3f}); "
      f"disagree on {(~agree).sum()}, which contain {(ee & ~agree).sum()} of the {ee.sum()} e5 errors")

print("\n" + "=" * 90 + "\nSample of 100 documents: expert, LLM and e5-large (Table 6)\n" + "=" * 90)
g = y[idx]; L_s = llm[idx]; E_s = e5[idx]
for name, p in [("expert", hum), ("LLM", L_s), ("e5-large", E_s)]:
    print(f"{name:9s} accuracy={accuracy_score(g, p):.2f} kappa={kappa(g, p):.3f} macro-F1={macro_f1(g, p):.3f} "
          f"F1 by area={dict(zip(AREAS, f1_score(g, p, labels=range(5), average=None, zero_division=0).round(2)))}")
print(f"kappa LLM-expert = {kappa(L_s, hum):.3f}")
b = int(((hum == g) & (E_s != g)).sum()); c = int(((hum != g) & (E_s == g)).sum())
print(f"McNemar expert vs e5-large: p = {binomtest(b, b + c, 0.5).pvalue:.2f}")
he = hum != g
for name, p in [("e5-large", E_s), ("LLM", L_s)]:
    pe = p != g
    tab = [[(he & pe).sum(), (he & ~pe).sum()], [(~he & pe).sum(), (~he & ~pe).sum()]]
    odds, pv = fisher_exact(tab)
    print(f"{name}: P(error | expert disagrees) = {pe[he].mean():.2f}, P(error | expert agrees) = {pe[~he].mean():.2f}, "
          f"odds ratio = {odds:.2f}, Fisher p = {pv:.4f}; shared errors {(he & pe).sum()}, same area {((p == hum) & he & pe).sum()}")
sec2 = ex["expert_second_area"].fillna("").astype(str).values
print(f"expert disagreements where the corpus label was the expert's second area: "
      f"{sum(sec2[i] == AREAS[g[i]] for i in np.where(he)[0])} of {he.sum()}")
conf = pd.to_numeric(ex["expert_confidence"], errors="coerce").values
print(f"expert mean confidence: agreements {conf[~he].mean():.2f}, disagreements {conf[he].mean():.2f}")
print(f"e5-large mean confidence: expert agrees {e5_conf[idx][~he].mean():.2f}, expert disagrees {e5_conf[idx][he].mean():.2f}")

FIG.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4))
for a, (p, t) in zip(ax, [(hum, "A  Independent expert"), (E_s, "B  multilingual-e5-large")]):
    sns.heatmap(confusion_matrix(g, p, labels=range(5)), annot=True, fmt="d", cmap="Blues", vmin=0, vmax=20,
                xticklabels=AREAS, yticklabels=AREAS, cbar=False, ax=a, annot_kws={"size": 9})
    a.set_title(t, fontsize=9, loc="left"); a.set_xlabel("Assigned area"); a.set_ylabel("Corpus label")
plt.tight_layout(); plt.savefig(FIG / "expert_vs_model_confusion.png", dpi=300, bbox_inches="tight")
