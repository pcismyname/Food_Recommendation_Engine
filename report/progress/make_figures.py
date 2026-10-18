"""Figures for the progress report, built only from saved result CSVs in baselines/cache/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "baselines" / "cache"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#3b6ea8", "#d9822b", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def bootstrap_ci(x, n=10_000, seed=42):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


# Figure 1: nDCG@10 with 95% bootstrap CI for every review-level model and floor
pq = pd.read_csv(CACHE / "per_query_review.csv")
rows = []
for name, g in pq.groupby("Model"):
    lo, hi = bootstrap_ci(g["nDCG@10"])
    rows.append((name, g["nDCG@10"].mean(), lo, hi))
df = pd.DataFrame(rows, columns=["model", "mean", "lo", "hi"]).sort_values("mean")
df["model"] = (df.model.str.replace("(v1)", "(proposal)", regex=False)
                 .str.replace("concat doc", "concatenated doc", regex=False)
                 .str.replace(" (LOO)", "", regex=False).str.replace(" (expected)", "", regex=False))
colors = [BLUE if m.startswith("BM25") else ORANGE if m.startswith("Dense") else GREY for m in df.model]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.barh(df.model, df["mean"], xerr=[df["mean"] - df.lo, df.hi - df["mean"]], color=colors, capsize=3)
for y, (m, h) in enumerate(zip(df["mean"], df.hi)):
    ax.text(h + 0.01, y, f"{m:.3f}", va="center", fontsize=8)
ax.set_xlim(0, 0.85)
ax.set_xlabel("nDCG@10 (mean over 100 queries, 95% bootstrap CI)")
ax.set_title("BM25 (blue), dense (orange) and simple baselines (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "v2_ndcg_ci.png", dpi=200); plt.close(fig)

# Figure 2: A3 volume control, full vs capped at 400 reviews
a3 = pd.read_csv(CACHE / "a3_volume_control_metrics_review.csv")
order = ["max", "mean", "top-3 mean", "top-5 mean", "top-10 mean"]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
for ax, model, col in zip(axes, ["BM25", "Dense"], [BLUE, ORANGE]):
    sub = a3[a3.model == model].pivot(index="aggregation", columns="run", values="nDCG@10").loc[order]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, sub["uncontrolled"], w, color=col, label="all reviews (406-2,281)")
    ax.bar(x + w / 2, sub["controlled_400"], w, color=col, alpha=0.45, hatch="//", label="capped at 400")
    for i, (a, b) in enumerate(zip(sub["uncontrolled"], sub["controlled_400"])):
        ax.text(i, max(a, b) + 0.01, f"{b - a:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, order, rotation=20); ax.set_ylim(0.5, 0.72); ax.set_title(model)
    ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("nDCG@10")
fig.suptitle("Change in nDCG@10 when every restaurant keeps 400 random reviews", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "a3_volume_control.png", dpi=200); plt.close(fig)

# Table numbers printed for the report: per-query score/count correlation
corr = pd.read_csv(CACHE / "a3_volume_control_correlations_review.csv")
corr = corr[corr["query"] != "__pooled__"]
corr["setting"] = np.where(corr.count_definition.eq("original_uncapped_reviews"), "capped_vs_original",
                           np.where(corr.run.str.endswith("uncontrolled"), "full", "capped_effective"))
print(corr[corr.setting != "capped_effective"]
      .groupby(["model", "aggregation", "setting"])[["pearson_score_vs_count", "spearman_score_vs_count"]]
      .mean().round(3).to_string())
print(df.round(3).to_string())
