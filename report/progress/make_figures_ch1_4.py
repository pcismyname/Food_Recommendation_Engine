"""Figures for Chapters 2 and 4 of the final report.

Built only from the raw datasets in datasets/ and the Thai label files in thai/.
Run from the repository root:  python report/progress/make_figures_ch1_4.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, GREEN = "#3b6ea8", "#d9822b", "#9a9a9a", "#4c9a5f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- RIRD (Figure 4.1)
rev = pd.read_csv(ROOT / "datasets/rir_data-main/reviews/50_restaurants_all_rates.csv")
rev["review_text"] = rev["review_text"].fillna("").astype(str)
pmd = pd.read_csv(ROOT / "datasets/rir_data-main/PMD.csv").drop_duplicates()
AGG = "If only Low or  High"  # double space is the real column name

counts = rev.groupby("business_id").size().sort_values(ascending=False)
rel_per_q = pmd.groupby("query")[AGG].sum()
words = rev["review_text"].str.split().str.len()

fig, ax = plt.subplots(2, 2, figsize=(8.4, 5.6))
a = ax[0, 0]
a.bar(np.arange(len(counts)), counts.values, color=BLUE, width=0.8)
a.axhline(400, color=ORANGE, ls="--", lw=1.2)
a.text(len(counts) - 1, 1300, "dashed line: 400-review cap\nused for the volume control", ha="right", color=ORANGE, fontsize=8)
a.set_xlabel("Restaurant (sorted by review count)"); a.set_ylabel("Reviews")
a.set_title(f"(a) Reviews per restaurant: {counts.min():,} to {counts.max():,}")

a = ax[0, 1]
a.hist(rel_per_q, bins=np.arange(0, rel_per_q.max() + 2) - 0.5, color=BLUE, edgecolor="white")
a.axvline(rel_per_q.mean(), color=ORANGE, ls="--", lw=1.2)
a.text(rel_per_q.mean() + 0.4, a.get_ylim()[1] * 0.9, f"mean {rel_per_q.mean():.1f}", color=ORANGE, fontsize=8)
a.set_xlabel("Relevant restaurants (of 50)"); a.set_ylabel("Queries")
a.set_title(f"(b) Relevant restaurants per query (n={len(rel_per_q)})")

a = ax[1, 0]
a.hist(words.clip(upper=600), bins=60, color=BLUE, edgecolor="white")
a.axvline(words.median(), color=ORANGE, ls="--", lw=1.2)
a.text(words.median() + 10, a.get_ylim()[1] * 0.9, f"median {words.median():.0f} words", color=ORANGE, fontsize=8)
a.set_xlabel("Review length in words (clipped at 600)"); a.set_ylabel("Reviews")
a.set_title("(c) Review length")

a = ax[1, 1]
st = rev["review_stars"].value_counts().sort_index()
a.bar(st.index.astype(int).astype(str), st.values / st.sum() * 100, color=BLUE)
a.set_xlabel("Star rating"); a.set_ylabel("% of reviews")
a.set_title("(d) Star ratings (metadata only)")
fig.tight_layout(); fig.savefig(OUT / "ch4_rird_eda.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Wongnai (Figure 4.2)
wg = pd.read_parquet(ROOT / "datasets/wongnai/train.parquet")
chars = wg["review_body"].astype(str).str.len()
fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
st = wg["star_rating"].value_counts().sort_index()
ax[0].bar(st.index.astype(str), st.values / st.sum() * 100, color=GREEN)
ax[0].set_xlabel("Star rating (0-4)"); ax[0].set_ylabel("% of reviews")
ax[0].set_title(f"(a) Star ratings, {len(wg):,} reviews")
ax[1].hist(chars.clip(upper=3000), bins=60, color=GREEN, edgecolor="white")
ax[1].axvline(chars.median(), color=ORANGE, ls="--", lw=1.2)
ax[1].text(chars.median() + 60, ax[1].get_ylim()[1] * 0.9, f"median {chars.median():.0f} characters",
           color=ORANGE, fontsize=8)
ax[1].set_xlabel("Review length in characters (clipped at 3,000)"); ax[1].set_ylabel("Reviews")
ax[1].set_title("(b) Review length")
fig.tight_layout(); fig.savefig(OUT / "ch4_wongnai_eda.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Thai test set (Figure 4.3)
q = pd.read_csv(ROOT / "thai/thai_queries.csv", encoding="utf-8-sig")
lab = pd.read_csv(ROOT / "thai/labels/label_sheet.csv", encoding="utf-8-sig").merge(q[["qid", "type"]], on="qid")
tags = q.assign(tag=q.taste_tags.str.split(";")).explode("tag").reset_index(drop=True)
tab = pd.crosstab(tags.tag, tags.type).reindex(columns=["single", "combo", "negation"], fill_value=0)
tab = tab.loc[tab.sum(axis=1).sort_values().index]
type_col = {"single": BLUE, "combo": ORANGE, "negation": GREY}

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
left = np.zeros(len(tab))
for t in tab.columns:
    ax[0].barh(tab.index, tab[t], left=left, color=type_col[t], label=f"{t} ({(q.type == t).sum()})")
    left += tab[t].values
ax[0].set_xlabel("Queries that contain the taste")
ax[0].set_title("(a) Taste coverage of the 24 frozen queries")
ax[0].legend(fontsize=7, loc="lower right", title="query type", title_fontsize=7)

lab_col = {0: "#d9d9d9", 1: "#9ecae1", 2: BLUE}
order = ["single", "combo", "negation", "all"]
share = pd.concat([pd.crosstab(lab.type, lab.label, normalize="index"),
                   lab.label.value_counts(normalize=True).rename("all").to_frame().T]).loc[order]
n = lab.groupby("type").size().reindex(order[:-1]).tolist() + [len(lab)]
left = np.zeros(len(order))
for l in [2, 1, 0]:
    vals = share[l].values * 100
    ax[1].barh(order, vals, left=left, color=lab_col[l], label={0: "0 not relevant", 1: "1 partial", 2: "2 relevant"}[l])
    for i, v in enumerate(vals):
        if v > 7:
            ax[1].text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                       color="white" if l == 2 else "black")
    left += vals
ax[1].set_yticks(range(len(order)), [f"{o}\n(n={k})" for o, k in zip(order, n)])
ax[1].invert_yaxis(); ax[1].set_xlim(0, 100); ax[1].set_xlabel("% of pooled pairs")
ax[1].set_title("(b) Human labels on the 452-pair pool")
ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "ch4_thai_testset.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Volume-bias simulation (Figure 2.2)
# Every restaurant draws unit scores from the SAME distribution, so none is more relevant.
rng = np.random.default_rng(42)
ns = np.array([406, 600, 800, 1000, 1300, 1600, 2000, 2281])
reps = 300
res = {"max": [], "top-10 mean": [], "mean": []}
for n_units in ns:
    s = rng.normal(size=(reps, n_units))
    s_sorted = np.sort(s, axis=1)
    res["max"].append(s_sorted[:, -1].mean())
    res["top-10 mean"].append(s_sorted[:, -10:].mean(axis=1).mean())
    res["mean"].append(s.mean(axis=1).mean())
fig, ax = plt.subplots(figsize=(5.6, 3.1))
for k, c, m in [("max", BLUE, "o"), ("top-10 mean", ORANGE, "s"), ("mean", GREY, "^")]:
    ax.plot(ns, res[k], color=c, marker=m, ms=4, label=k)
ax.set_xlabel("Number of reviews of a restaurant (RIRD range 406-2,281)")
ax.set_ylabel("Expected restaurant score\n(standardised unit scores)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Same relevance, more reviews: max and top-k scores still rise", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "ch2_volume_simulation.png", dpi=200); plt.close(fig)

print("RIRD reviews/restaurant", counts.min(), counts.max(), "median words", words.median(),
      "relevant/query mean", round(rel_per_q.mean(), 2), "min", rel_per_q.min(), "max", rel_per_q.max())
print("Wongnai", len(wg), "median chars", chars.median(), "stars %", (st / st.sum() * 100).round(1).to_dict())
print("RIRD stars %", (rev.review_stars.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print(share.round(3)); print(tab)
print({k: np.round(v, 3).tolist() for k, v in res.items()})

# ---------------------------------------------------------------- Thai results (for Chapter 6, Sam)
summ = pd.read_csv(ROOT / "thai/results/thai_scores_summary.csv", encoding="utf-8-sig")
pq = pd.read_csv(ROOT / "thai/results/thai_scores_per_query.csv", encoding="utf-8-sig")
summ = summ[summ.metric == "nDCG@10"].copy()
summ["scope"] = summ.scope.str.replace("type=", "", regex=False)
scopes = ["all", "single", "combo", "negation"]
summ = summ.set_index("scope").loc[scopes]
nq = {"all": 24, "single": 11, "combo": 8, "negation": 5}
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(len(scopes)); w = 0.38
for off, sysn, col in [(-w / 2, "bm25", BLUE), (w / 2, "dense", ORANGE)]:
    m = summ[f"{sysn}_mean"]; lo = summ[f"{sysn}_ci_lo"]; hi = summ[f"{sysn}_ci_hi"]
    ax[0].bar(x + off, m, w, yerr=[m - lo, hi - m], color=col, capsize=3,
              label="BM25" if sysn == "bm25" else "Dense")
    for i, v in enumerate(m):
        ax[0].text(x[i] + off, 0.03, f"{v:.2f}", ha="center", fontsize=7, color="white", rotation=90)
ax[0].set_xticks(x, [f"{s}\n(n={nq[s]})" for s in scopes]); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("nDCG@10 (95% bootstrap CI)"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) Thai sentence retrieval by query type")
wide = pq.pivot(index="qid", columns="system", values="nDCG@10")
wide["diff"] = wide.dense - wide.bm25
wide["type"] = pq[pq.system == "bm25"].set_index("qid")["type"]
wide = wide.sort_values("diff")
ax[1].bar(range(len(wide)), wide["diff"], color=[type_col[t] for t in wide["type"]])
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xticks(range(len(wide)), wide.index, rotation=90, fontsize=6)
ax[1].set_ylabel("Dense minus BM25 nDCG@10")
ax[1].set_title(f"(b) Per query: dense better {(wide['diff'] > 0).sum()}, "
                f"BM25 better {(wide['diff'] < 0).sum()}, tie {(wide['diff'] == 0).sum()}", fontsize=8.5)
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=type_col[t], label=t) for t in ["single", "combo", "negation"]],
             fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch6_thai_results.png", dpi=200); plt.close(fig)
