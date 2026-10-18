"""
score_labels.py — C7: agreement + Thai nDCG@10 / P@10 for BM25 vs. dense
=========================================================================
Inputs (from thai/build_label_pool.py, labels filled in by hand):
    thai/labels/label_sheet.csv              pair_id, qid, ..., label (0/1/2), note
    thai/labels/pool_key.csv                 hidden key: which system retrieved each pair
    thai/labels/label_sheet_annotator2.csv   optional second annotator (C5)
    thai/thai_queries.csv                    for the query `type` breakdown

Metrics per query per system (graded gains 0/1/2):
    nDCG@10        DCG of the system's top 10; ideal DCG built from ALL labelled pooled
                   sentences of that query (standard pooled evaluation). 0 if the query
                   has no relevant sentence in the pool.
    P@10_lenient   share of the top 10 with label >= 1
    P@10_strict    share of the top 10 with label == 2
Also the number of relevant (>=1 and ==2) sentences in each query's pool.

Over queries: mean with 95% percentile bootstrap CI (10,000 resamples, seed 42), paired
randomization (sign-flip, two-sided) test dense vs. bm25, Holm correction across the
three metrics. Breakdown by query type (single / combo / negation) is descriptive only
(means + CIs, no tests: too few queries per type).

Agreement (if the annotator-2 sheet has labels): % exact agreement, Cohen's kappa
(unweighted and linear-weighted) on overlapping labelled pair_ids.

KNOWN LIMITATION (pooling bias): the pool is only the top 10 of two systems. Sentences
neither system retrieved are never judged, so the ideal DCG is relative to the pool,
not to the whole corpus. nDCG here measures how close each system gets to the best
ordering of what the two systems jointly found; it overstates absolute quality and can
favour whichever system contributed more relevant sentences to the pool. Because every
sentence in each system's top 10 is in the pool, there are no unjudged documents in
the ranked lists themselves (asserted below).

Usage:
    python thai/score_labels.py                 # all queries must be fully labelled
    python thai/score_labels.py --partial       # score only fully-labelled queries
    python thai/score_labels.py --labels-dir thai/labels --results-dir thai/results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SYSTEMS = ("bm25", "dense")
K = 10
SEED = 42
N_RESAMPLES = 10_000
METRICS = ["nDCG@10", "P@10_lenient", "P@10_strict"]


# ─────────────────────────────────────────────
# Stats (same conventions as baselines/baseline_models_v2.ipynb)
# ─────────────────────────────────────────────

def bootstrap_ci(x, n=N_RESAMPLES, seed=SEED):
    """95% percentile bootstrap CI of the mean over queries."""
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def paired_randomization_p(a, b, n=N_RESAMPLES, seed=SEED):
    """Two-sided paired randomization (sign-flip) test on per-query differences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1)


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def cohen_kappa(a, b, weights=None, categories=(0, 1, 2)):
    """Cohen's kappa; weights=None (unweighted) or 'linear'."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    cats = list(categories)
    m = len(cats)
    pos = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((m, m))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((m, m))
    w = (i != j).astype(float) if weights is None else np.abs(i - j) / (m - 1)
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return 1.0 - (w * obs).sum() / denom


# ─────────────────────────────────────────────
# Loading / validation
# ─────────────────────────────────────────────

def parse_labels(sheet: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add integer column `y` (NaN if unlabelled); exit on any invalid label."""
    raw = sheet["label"].fillna("").astype(str).str.strip()
    num = pd.to_numeric(raw.where(raw != ""), errors="coerce")
    bad = (raw != "") & ~num.isin([0, 1, 2])
    if bad.any():
        rows = sheet.loc[bad, ["pair_id", "label"]].head(20).to_string(index=False)
        sys.exit(f"ERROR: {bad.sum()} invalid label(s) in {name} (must be 0, 1 or 2):\n{rows}")
    sheet = sheet.copy()
    sheet["y"] = num
    return sheet


def read_sheet(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score Thai labels (C7).")
    p.add_argument("--labels-dir", default=str(REPO / "thai/labels"))
    p.add_argument("--queries", default=str(REPO / "thai/thai_queries.csv"))
    p.add_argument("--results-dir", default=str(REPO / "thai/results"))
    p.add_argument("--partial", action="store_true",
                   help="score only queries whose pool is fully labelled")
    return p.parse_args(argv)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def dcg(gains):
    gains = np.asarray(gains, dtype=float)[:K]
    return float((gains / np.log2(np.arange(2, len(gains) + 2))).sum())


def per_query_scores(df: pd.DataFrame) -> pd.DataFrame:
    """df: pool_key joined with labels (y), all rows of scored queries labelled."""
    rows = []
    for qid, g in df.groupby("qid"):
        ideal = dcg(np.sort(g["y"].to_numpy())[::-1])
        n_rel_len = int((g["y"] >= 1).sum())
        n_rel_str = int((g["y"] == 2).sum())
        for s in SYSTEMS:
            ranked = g[g[f"in_{s}"] == 1].sort_values(f"{s}_rank")
            assert ranked["y"].notna().all(), f"unjudged sentence in {s} top-{K} for {qid}"
            assert len(ranked) <= K
            y = ranked["y"].to_numpy()
            rows.append({
                "qid": qid, "system": s, "n_retrieved": len(ranked),
                "nDCG@10": dcg(y) / ideal if ideal > 0 else 0.0,
                "P@10_lenient": float((y >= 1).sum()) / K,
                "P@10_strict": float((y == 2).sum()) / K,
                "pool_size": len(g), "pool_rel_lenient": n_rel_len,
                "pool_rel_strict": n_rel_str,
            })
    return pd.DataFrame(rows)


def summarise(pq: pd.DataFrame, scope: str, test: bool) -> pd.DataFrame:
    rows = []
    wide = {m: pq.pivot(index="qid", columns="system", values=m) for m in METRICS}
    for m in METRICS:
        w = wide[m]
        row = {"scope": scope, "metric": m, "n_queries": len(w)}
        for s in SYSTEMS:
            lo, hi = bootstrap_ci(w[s])
            row.update({f"{s}_mean": w[s].mean(), f"{s}_ci_lo": lo, f"{s}_ci_hi": hi})
        d = w["dense"] - w["bm25"]
        lo, hi = bootstrap_ci(d)
        row.update({"diff_dense_minus_bm25": d.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "p_randomization": paired_randomization_p(w["dense"], w["bm25"])
                    if test else np.nan})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_randomization"]) if test else np.nan
    return out


def agreement(sheet1: pd.DataFrame, a2_path: Path):
    if not a2_path.exists():
        return None, "no annotator-2 sheet found"
    a2 = parse_labels(read_sheet(a2_path), a2_path.name)
    m = sheet1[["pair_id", "text", "y"]].merge(a2[["pair_id", "text", "y"]],
                                               on="pair_id", suffixes=("_1", "_2"))
    if (m["text_1"] != m["text_2"]).any():
        sys.exit("ERROR: annotator-2 sheet text does not match label_sheet.csv for some "
                 "pair_ids (rows edited or sheets from different pools?)")
    m = m.dropna(subset=["y_1", "y_2"])
    if m.empty:
        return None, "annotator-2 sheet has no overlapping labelled pairs yet"
    a, b = m["y_1"].astype(int), m["y_2"].astype(int)
    res = {
        "n_pairs": len(m),
        "exact_agreement": float((a == b).mean()),
        "agreement_lenient_binary": float(((a >= 1) == (b >= 1)).mean()),
        "kappa_unweighted": cohen_kappa(a, b),
        "kappa_linear": cohen_kappa(a, b, weights="linear"),
    }
    try:  # cross-check against sklearn when available
        from sklearn.metrics import cohen_kappa_score
        for k, w in (("kappa_unweighted", None), ("kappa_linear", "linear")):
            ref = cohen_kappa_score(a, b, weights=w, labels=[0, 1, 2])
            if not (np.isnan(ref) and np.isnan(res[k])):
                assert abs(ref - res[k]) < 1e-9, (k, ref, res[k])
    except ImportError:
        pass
    return res, None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    labels_dir, results_dir = Path(args.labels_dir), Path(args.results_dir)

    sheet = parse_labels(read_sheet(labels_dir / "label_sheet.csv"), "label_sheet.csv")
    key = pd.read_csv(labels_dir / "pool_key.csv", dtype={"pair_id": str, "qid": str,
                      "sent_id": str, "review_id": str}, encoding="utf-8-sig")
    queries = pd.read_csv(args.queries, dtype=str, encoding="utf-8-sig")

    if set(sheet["pair_id"]) != set(key["pair_id"]) or len(sheet) != len(key):
        sys.exit("ERROR: label_sheet.csv and pool_key.csv have different pair_ids "
                 "(rows deleted/added?). Restore the sheet from build_label_pool.py output.")
    df = key.merge(sheet[["pair_id", "qid", "y"]], on="pair_id", suffixes=("", "_sheet"))
    if (df["qid"] != df["qid_sheet"]).any():
        sys.exit("ERROR: qid mismatch between label_sheet.csv and pool_key.csv.")

    # Labelling progress
    n_un = int(df["y"].isna().sum())
    prog = df.groupby("qid")["y"].agg(labelled="count", total="size")
    complete = prog.index[prog["labelled"] == prog["total"]].tolist()
    incomplete = prog.index[prog["labelled"] < prog["total"]].tolist()
    print(f"Labels: {len(df) - n_un}/{len(df)} pairs labelled ({n_un} unlabelled); "
          f"{len(complete)}/{len(prog)} queries complete.")
    if incomplete:
        print("Incomplete queries: " + ", ".join(
            f"{q} ({prog.at[q, 'labelled']}/{prog.at[q, 'total']})" for q in incomplete))
        if not args.partial:
            sys.exit("Not all queries are fully labelled. Finish labelling, or re-run with "
                     "--partial to score only the complete queries.")
        print(f"--partial: scoring ONLY these {len(complete)} complete queries: "
              f"{', '.join(complete)}")
    if not complete:
        sys.exit("No fully-labelled query to score yet.")

    df = df[df["qid"].isin(complete)]
    pq = per_query_scores(df).merge(queries[["qid", "type"]], on="qid", how="left")
    pq["type"] = pq["type"].fillna("unknown")

    summary = [summarise(pq, "all", test=len(complete) >= 2)]
    for t, g in pq.groupby("type"):
        summary.append(summarise(g, f"type={t}", test=False))
    summary = pd.concat(summary, ignore_index=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    pq.to_csv(results_dir / "thai_scores_per_query.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results_dir / "thai_scores_summary.csv", index=False, encoding="utf-8-sig")

    agr, why = agreement(sheet, labels_dir / "label_sheet_annotator2.csv")
    if agr:
        pd.DataFrame([agr]).to_csv(results_dir / "thai_agreement.csv", index=False)

    # Readable summary
    pd.set_option("display.width", 140)
    print(f"\n=== Thai sentence retrieval, {len(complete)} queries, pool depth {K} "
          f"(nDCG relative to pool) ===")
    for scope, g in summary.groupby("scope", sort=False):
        print(f"\n[{scope}]  n_queries={int(g['n_queries'].iloc[0])}")
        for _, r in g.iterrows():
            line = (f"  {r['metric']:<13} bm25 {r['bm25_mean']:.3f} "
                    f"[{r['bm25_ci_lo']:.3f}, {r['bm25_ci_hi']:.3f}]   "
                    f"dense {r['dense_mean']:.3f} [{r['dense_ci_lo']:.3f}, "
                    f"{r['dense_ci_hi']:.3f}]   diff {r['diff_dense_minus_bm25']:+.3f} "
                    f"[{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]")
            if not np.isnan(r["p_randomization"]):
                line += f"   p={r['p_randomization']:.4f} p_holm={r['p_holm']:.4f}"
            print(line)
    pool_stats = pq.drop_duplicates("qid")
    print(f"\nPool: mean {pool_stats['pool_size'].mean():.1f} sentences/query; relevant "
          f"(>=1) mean {pool_stats['pool_rel_lenient'].mean():.1f}, (==2) mean "
          f"{pool_stats['pool_rel_strict'].mean():.1f}; queries with no relevant "
          f"sentence in pool: {int((pool_stats['pool_rel_lenient'] == 0).sum())}")
    if agr:
        print(f"\nAgreement (annotator 1 vs 2, n={agr['n_pairs']} pairs): exact "
              f"{agr['exact_agreement']:.1%}, lenient-binary "
              f"{agr['agreement_lenient_binary']:.1%}, kappa {agr['kappa_unweighted']:.3f}, "
              f"linear-weighted kappa {agr['kappa_linear']:.3f}")
    else:
        print(f"\nAgreement: skipped ({why})")
    print(f"\nWrote {results_dir / 'thai_scores_per_query.csv'}\n      "
          f"{results_dir / 'thai_scores_summary.csv'}"
          + (f"\n      {results_dir / 'thai_agreement.csv'}" if agr else ""))


if __name__ == "__main__":
    main()
