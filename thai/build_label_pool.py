"""
build_label_pool.py — C3: pool BM25 + dense top-10 sentences into a blind label sheet
=====================================================================================
Input : thai/results/topk_n10000_s42.csv  (written by the Thai retrieval notebook)
        columns: qid, query_th, system, rank, sent_id, review_id, sent_idx, text,
                 score, star_rating, n_reviews, seed      (system in {bm25, dense})
        thai/thai_queries.csv                 (for query_en)

Output (in --out-dir, default thai/labels):
    label_sheet.csv             blind sheet for the annotator (utf-8-sig, opens in Excel /
                                Google Sheets with Thai intact). Grouped by qid, order
                                within each query shuffled so system / rank can't be
                                inferred. Columns: pair_id, qid, query_th, query_en, text,
                                label, note.
    pool_key.csv                hidden key: which system(s) retrieved each pair, at what
                                rank and score. Do NOT show to the annotator.
    label_sheet_annotator2.csv  seeded stratified sample (--second-n pairs, spread across
                                queries) for a second annotator (C5). Same blind columns.

The pool for a query is the de-duplicated union of sent_ids in both systems' top 10.

Usage:
    python thai/build_label_pool.py
    python thai/build_label_pool.py --topk path/to/topk.csv --out-dir thai/labels \
        --seed 42 --second-n 60 [--force]

An existing label_sheet.csv that already contains any label is never overwritten
unless --force is given (protects hand labels).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TOPK_COLS = ["qid", "query_th", "system", "rank", "sent_id", "review_id", "sent_idx",
             "text", "score", "star_rating", "n_reviews", "seed"]
SYSTEMS = ("bm25", "dense")
K = 10
SHEET_COLS = ["pair_id", "qid", "query_th", "query_en", "text", "label", "note"]
KEY_COLS = ["pair_id", "qid", "sent_id", "review_id", "sent_idx",
            "in_bm25", "bm25_rank", "bm25_score",
            "in_dense", "dense_rank", "dense_score", "star_rating"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--topk", default=str(REPO / "thai/results/topk_n10000_s42.csv"))
    p.add_argument("--queries", default=str(REPO / "thai/thai_queries.csv"))
    p.add_argument("--out-dir", default=str(REPO / "thai/labels"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--second-n", type=int, default=60,
                   help="pairs in the second-annotator sample (C5)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a label_sheet.csv that already has labels")
    return p.parse_args(argv)


def sheet_has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return "label" in df.columns and (df["label"].str.strip() != "").any()


def load_topk(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sent_id": str, "review_id": str, "qid": str},
                     encoding="utf-8-sig")
    missing = [c for c in TOPK_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing columns {missing}")
    bad = set(df["system"]) - set(SYSTEMS)
    if bad:
        sys.exit(f"ERROR: unexpected system values {bad}; expected {SYSTEMS}")
    df = df[df["rank"] <= K].copy()
    # A sentence should appear once per (query, system); keep its best rank if not.
    df = (df.sort_values(["qid", "system", "rank"])
            .drop_duplicates(["qid", "system", "sent_id"], keep="first"))
    return df


def build_pool(topk: pd.DataFrame, queries: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One row per (qid, sent_id), with per-system rank/score; shuffled within query."""
    base = (topk.sort_values(["qid", "system", "rank"])
                .drop_duplicates(["qid", "sent_id"], keep="first")
                [["qid", "query_th", "sent_id", "review_id", "sent_idx", "text",
                  "star_rating"]])
    for s in SYSTEMS:
        sub = topk[topk["system"] == s][["qid", "sent_id", "rank", "score"]]
        sub = sub.rename(columns={"rank": f"{s}_rank", "score": f"{s}_score"})
        base = base.merge(sub, on=["qid", "sent_id"], how="left")
        base[f"in_{s}"] = base[f"{s}_rank"].notna().astype(int)
        base[f"{s}_rank"] = base[f"{s}_rank"].astype("Int64")

    base = base.merge(queries[["qid", "query_en"]], on="qid", how="left")
    if base["query_en"].isna().any():
        missing = sorted(base.loc[base["query_en"].isna(), "qid"].unique())
        print(f"WARNING: qids not in thai_queries.csv: {missing}")
        base["query_en"] = base["query_en"].fillna("")

    # Shuffle order within each query (seeded) so system/rank isn't inferable;
    # keep queries grouped and ordered by qid for labelling one query at a time.
    rng = np.random.default_rng(seed)
    base["_rand"] = rng.permutation(len(base))
    base = base.sort_values(["qid", "_rand"]).drop(columns="_rand").reset_index(drop=True)
    base.insert(0, "pair_id", [f"P{i:04d}" for i in range(1, len(base) + 1)])
    return base


def stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Spread n pairs as evenly as possible across queries (seeded)."""
    rng = np.random.default_rng(seed + 1)
    sizes = pool.groupby("qid").size()
    n = min(n, len(pool))
    qids = list(sizes.index)
    quota = {q: 0 for q in qids}
    remaining = n
    # Round-robin in a random query order, respecting each query's pool size.
    while remaining > 0:
        order = [qids[i] for i in rng.permutation(len(qids))]
        progressed = False
        for q in order:
            if remaining == 0:
                break
            if quota[q] < sizes[q]:
                quota[q] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    picks = []
    for q, g in pool.groupby("qid"):
        if quota[q]:
            picks.append(g.iloc[rng.choice(len(g), size=quota[q], replace=False)])
    return pd.concat(picks).sort_values("pair_id")


def blind(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["pair_id", "qid", "query_th", "query_en", "text"]].copy()
    out["label"] = ""
    out["note"] = ""
    return out[SHEET_COLS]


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "label_sheet.csv"
    a2_path = out_dir / "label_sheet_annotator2.csv"

    for p in (sheet_path, a2_path):
        if sheet_has_labels(p) and not args.force:
            sys.exit(f"ERROR: {p} already contains labels. Refusing to overwrite "
                     f"(re-run with --force if you really mean it).")

    topk = load_topk(args.topk)
    queries = pd.read_csv(args.queries, dtype=str, encoding="utf-8-sig")
    pool = build_pool(topk, queries, args.seed)

    blind(pool).to_csv(sheet_path, index=False, encoding="utf-8-sig")
    pool[KEY_COLS].to_csv(out_dir / "pool_key.csv", index=False, encoding="utf-8-sig")
    sample = stratified_sample(pool, args.second_n, args.seed)
    blind(sample).to_csv(a2_path, index=False, encoding="utf-8-sig")

    # Report
    flags = pool.assign(
        both=(pool["in_bm25"] == 1) & (pool["in_dense"] == 1),
        bm25_only=(pool["in_bm25"] == 1) & (pool["in_dense"] == 0),
        dense_only=(pool["in_bm25"] == 0) & (pool["in_dense"] == 1))
    per_q = flags.groupby("qid").agg(pool=("sent_id", "size"), both=("both", "sum"),
                                     bm25_only=("bm25_only", "sum"),
                                     dense_only=("dense_only", "sum"))
    print("Pool size per query (overlap = sentences found by both systems):")
    print(per_q[["pool", "both", "bm25_only", "dense_only"]].to_string())
    print(f"\nTotal pairs: {len(pool)} over {pool['qid'].nunique()} queries "
          f"(mean {len(pool) / max(pool['qid'].nunique(), 1):.1f} per query)")
    print(f"Second-annotator sample: {len(sample)} pairs over "
          f"{sample['qid'].nunique()} queries")
    print(f"Wrote {sheet_path}\n      {out_dir / 'pool_key.csv'}  (hidden key)\n      {a2_path}")


if __name__ == "__main__":
    main()
