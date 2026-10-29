"""
Word-level aggregation of SHAP attributions (Methods: interpretability analysis; Results: interpretability and
qualitative error analysis; Figs 2 and 3).

The SHAP values are produced by notebooks/MTSK_modelo_final_SHAP.ipynb, which saves
`shap_values.pkl` and `test_predictions.csv` for the 56 test documents of fold 0.
The pickle is not distributed in this repository because it contains the tokenised
text of copyrighted abstracts; regenerate it with the notebook and pass its path.

Usage  : python scripts/shap_word_level.py --shap path/to/shap_values.pkl \
                                           --pred path/to/test_predictions.csv
Outputs: results/final_model/shap_word_level_by_class.csv,
         figures/shap_word_level_by_class.png, figures/shap_cases.png
"""
import argparse
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import AREAS, FIG, RES

STOP = set("""
de la el los las del al a en y o u e que se su sus un una uno unos unas por para con sin sobre entre como más mas este esta
estos estas ese esa eso esto lo le les es son ser fue han ha hay nos no ni tanto también tambien cual cuales cuando donde muy
mismo misma desde hasta durante tras ante bajo según segun cada otro otra otros otras dos tres así asi pero sino porque
do da dos das na nos nas um uma uns umas com em ao aos à às pelo pela pelos pelas é são foi seu sua seus suas isso isto ou os as o
neste nesta desse dessa deste desta
the of and to in on for with by an a is are was were be been this that these those as at from or it its their which we our
into how what about between through during also not
""".split())


def to_words(expl):
    """Merge SentencePiece subtokens into words (a piece ending in whitespace closes a word)."""
    out, cur, val = [], "", np.zeros(expl.values.shape[1])
    for tok, v in zip(expl.data, expl.values):
        tok = str(tok)
        if not tok:
            continue
        cur += tok; val = val + v
        if tok.endswith((" ", "\n")):
            out.append((cur.strip(), val)); cur, val = "", np.zeros_like(val)
    if cur:
        out.append((cur.strip(), val))
    return out


def norm(w):
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w.lower().strip())
    return w[1:] if len(w) > 3 and w[0] == w[1] and w[0] != "l" else w   # masker artefact


def main(shap_path, pred_path, min_share=0.25, top=10):
    sv = pickle.load(open(shap_path, "rb"))
    pred = pd.read_csv(pred_path)
    rows, mass = [], {"content": 0.0, "stop": 0.0}
    for d in range(len(pred)):
        words = to_words(sv[d])
        if words and norm(words[0][0]) in {"query", "query:"}:
            words = words[1:]
        seen = {}
        for w, v in words:
            n = norm(w)
            kind = "stop" if (not n or n in STOP or len(n) <= 2) else "content"
            mass[kind] += np.abs(v).sum()
            if kind == "content":
                seen[n] = seen.get(n, 0) + v
        rows += [{"doc": d, "word": n, **{AREAS[c]: v[c] for c in range(5)}} for n, v in seen.items()]
    W = pd.DataFrame(rows).merge(pred[["doc", "true"]], on="doc")
    tot = sum(mass.values())
    print(f"attribution share: content words {mass['content'] / tot:.1%}, stop words {mass['stop'] / tot:.1%}")

    res = []
    for c in AREAS:
        n_in, n_out = (pred["true"] == c).sum(), (pred["true"] != c).sum()
        gin = W[W["true"] == c].groupby("word")[c].agg(["sum", "count"])
        gout = W[W["true"] != c].groupby("word")[c].sum()
        t = gin.join(gout.rename("sum_out"), how="left").fillna(0)
        t = t[t["count"] >= max(2, int(np.ceil(min_share * n_in)))]
        res.append(pd.DataFrame({"class": c, "n_class_docs": n_in, "word": t.index, "docs_with_word": t["count"].astype(int),
                                 "mean_support_in_class": t["sum"] / n_in,
                                 "mean_support_other_classes": t["sum_out"] / n_out})
                   .assign(distinctiveness=lambda x: x.mean_support_in_class - x.mean_support_other_classes)
                   .sort_values("mean_support_in_class", ascending=False))
    R = pd.concat(res)
    R.to_csv(RES / "final_model" / "shap_word_level_by_class.csv", index=False)

    FIG.mkdir(exist_ok=True)
    names = {"T1": "T1 · Teacher education", "T2": "T2 · Teacher educators", "T3": "T3 · Topics and stages",
             "T4": "T4 · Theoretical development", "T5": "T5 · Extensions"}
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 6.4))
    axes = axes.ravel(); axes[5].axis("off")
    for ax, c in zip(axes, AREAS):
        t = R[R["class"] == c].head(top)[::-1]
        ax.barh(t["word"], t["mean_support_in_class"], color="#2E86C1")
        ax.set_title(f"{names[c]}\n(n = {int(t['n_class_docs'].iloc[0])} test documents)", fontsize=8)
        ax.set_xlabel("Mean SHAP contribution", fontsize=8); ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(FIG / "shap_word_level_by_class.png", dpi=300, bbox_inches="tight")

    def panel(ax, d, cls, title, n=12):
        w = W[W["doc"] == d][["word", cls]]
        w = w.reindex(w[cls].abs().sort_values(ascending=False).index).head(n)[::-1]
        ax.barh(w["word"], w[cls], color=["#C0392B" if v > 0 else "#2E86C1" for v in w[cls]])
        ax.axvline(0, color="black", lw=0.8)
        span = w[cls].abs().max()
        for yy, v in enumerate(w[cls]):
            ax.text(v + (0.02 if v > 0 else -0.02) * span, yy, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=7)
        lo, hi = min(0, w[cls].min()), max(0, w[cls].max()); r = hi - lo
        ax.set_xlim(lo - 0.18 * r, hi + 0.24 * r)
        ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel(f"SHAP contribution toward {cls}", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2))
    panel(axes[0], 1, "T2", "A  Document 1 (true T1, predicted T2)")
    panel(axes[1], 38, "T3", "B  Document 38 (true T4, predicted T3)")
    plt.tight_layout(); plt.savefig(FIG / "shap_cases.png", dpi=300, bbox_inches="tight")
    print("Saved CSV and figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", required=True)
    ap.add_argument("--pred", default=str(RES / "final_model" / "test_predictions_fold0.csv"))
    a = ap.parse_args()
    main(a.shap, a.pred)
