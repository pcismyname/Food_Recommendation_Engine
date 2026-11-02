"""b2_export_support_sheet.py — Build the human-labelling sheet for B2.

For the Dense | top-10 mean sentence-level run, this script exports the
shown explanation sentences for a sample of queries × restaurants into a
blind CSV sheet for two annotators to label independently.

Usage
-----
    python baselines/b2_export_support_sheet.py

Outputs (written to baselines/cache/)
--------------------------------------
    b2_support_sheet.csv          — full sheet for annotator 1 (Sam)
    b2_support_sheet_ann2.csv     — 40-pair subset for annotator 2 (Ton)
    b2_support_key.csv            — hidden key (restaurant rank, score); DO NOT share

Labelling scale (see thai/annotation_guidelines.md for full guide)
-------------------------------------------------------------------
    0 = The sentence does NOT support the query for this restaurant.
    1 = The sentence partially supports the query / weakly relevant.
    2 = The sentence clearly supports why this restaurant matches the query.

Instructions for annotators
-----------------------------
1. Open b2_support_sheet.csv (or b2_support_sheet_ann2.csv for annotator 2).
2. For each row read `query` and `sentence_text`, then fill `label` (0/1/2).
   The `restaurant_id` column is intentionally included — it tells you which
   restaurant this sentence came from, but NOT its rank.
3. Optionally add a free-text note in the `note` column.
4. Save as CSV (UTF-8) with the same filename.
5. Compute agreement with: python thai/score_labels.py --b2
   (or load both CSVs and compute Cohen's kappa manually).
"""

import sys
import csv
import random
import itertools
from pathlib import Path
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_MODEL = "Dense"
TARGET_AGG   = "top-10 mean"
N_QUERIES    = 25     # sample this many queries (or all if fewer available)
K_REST       = 3      # top-K restaurants per query
K_EV         = 5      # top-K evidence sentences per (query, restaurant) to export
                      # (show top-5 of the top-10; avoids an overwhelming sheet)
N_ANN2       = 40     # number of pairs for annotator-2 subset sheet
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

HERE  = Path(__file__).resolve().parent
ROOT  = HERE.parent
CACHE = HERE / "cache"


def _load_csv(name: str) -> list[dict]:
    """Load a CSV from cache/sentence/ or cache/, return list of dicts."""
    for p in [CACHE / "sentence" / name, CACHE / name]:
        if p.exists():
            print(f"  Loaded: {p}")
            with open(p, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(f"{name} not found in {CACHE}")


def main():
    print("=== B2 Support Sheet Builder ===\n")

    # ── Load evidence ──────────────────────────────────────────────────────
    evidence_rows = _load_csv("evidence_sentence.csv")
    ranking_rows  = _load_csv("rankings_sentence.csv")

    # Filter to target model/agg
    evidence = [r for r in evidence_rows
                if r["model"] == TARGET_MODEL and r["aggregation"] == TARGET_AGG]
    rankings = [r for r in ranking_rows
                if r["model"] == TARGET_MODEL and r["aggregation"] == TARGET_AGG]

    print(f"Evidence rows ({TARGET_MODEL}|{TARGET_AGG}): {len(evidence):,}")
    print(f"Ranking rows:  {len(rankings):,}")

    # ── Get ordered queries from rankings ─────────────────────────────────
    all_queries = list(dict.fromkeys(r["query"] for r in rankings))  # preserve order
    print(f"\nTotal queries: {len(all_queries)}")

    # Sample N_QUERIES
    if len(all_queries) > N_QUERIES:
        sampled_queries = random.sample(all_queries, N_QUERIES)
    else:
        sampled_queries = all_queries
    sampled_queries_set = set(sampled_queries)
    print(f"Sampled queries: {len(sampled_queries)}")

    # ── Build ranking lookup: query → [restaurant_id, ...] ordered by rank ─
    rank_lookup: dict[str, list[str]] = defaultdict(list)
    for r in sorted(rankings, key=lambda x: (x["query"], int(x["restaurant_rank"]))):
        if r["query"] in sampled_queries_set:
            rank_lookup[r["query"]].append(r["restaurant_id"])

    # ── Build evidence lookup: (query, restaurant) → [(evidence_rank, score, text), ...]
    ev_lookup: dict[tuple, list] = defaultdict(list)
    for e in evidence:
        if e["query"] not in sampled_queries_set:
            continue
        key = (e["query"], e["restaurant_id"])
        ev_lookup[key].append((
            int(e["evidence_rank"]),
            float(e["unit_score"]),
            e["text"],
            e.get("unit_id", ""),
        ))

    # Sort by evidence_rank ascending (rank 1 = highest score)
    for key in ev_lookup:
        ev_lookup[key].sort(key=lambda x: x[0])

    # ── Build sheet rows ────────────────────────────────────────────────────
    sheet_rows = []   # for annotation
    key_rows   = []   # hidden key (restaurant rank, scores)
    pair_id    = 0

    for q in sampled_queries:
        top_restaurants = rank_lookup[q][:K_REST]
        for rank_pos, rest_id in enumerate(top_restaurants, start=1):
            sentences = ev_lookup.get((q, rest_id), [])[:K_EV]
            for ev_rank, ev_score, text, unit_id in sentences:
                pair_id += 1
                sheet_rows.append({
                    "pair_id":         pair_id,
                    "query":           q,
                    "restaurant_id":   rest_id,
                    "sentence_text":   text,
                    "label":           "",   # annotator fills this in
                    "note":            "",
                })
                key_rows.append({
                    "pair_id":          pair_id,
                    "query":            q,
                    "restaurant_id":    rest_id,
                    "restaurant_rank":  rank_pos,   # rank among 50 restaurants
                    "evidence_rank":    ev_rank,
                    "unit_score":       ev_score,
                    "unit_id":          unit_id,
                    "sentence_text":    text,
                })

    print(f"\nTotal (query, restaurant, sentence) pairs: {len(sheet_rows)}")
    print(f"  = {len(sampled_queries)} queries × up to {K_REST} restaurants "
          f"× up to {K_EV} sentences")

    # ── Annotator-2 subset: spread across all sampled queries ──────────────
    # Pick N_ANN2 pairs, ensuring at least 1 per query where possible
    queries_in_sheet = list(dict.fromkeys(r["query"] for r in sheet_rows))
    ann2_pairs: list[dict] = []

    # One pair per query (first in sheet order)
    seen_q: set = set()
    for row in sheet_rows:
        if row["query"] not in seen_q:
            ann2_pairs.append(row)
            seen_q.add(row["query"])
        if len(ann2_pairs) >= N_ANN2:
            break

    # Top-up with random pairs if needed
    remaining = [r for r in sheet_rows if r not in ann2_pairs]
    if len(ann2_pairs) < N_ANN2 and remaining:
        extra = random.sample(remaining, min(N_ANN2 - len(ann2_pairs), len(remaining)))
        ann2_pairs.extend(extra)

    ann2_pairs.sort(key=lambda x: x["pair_id"])
    print(f"Annotator-2 subset: {len(ann2_pairs)} pairs")

    # ── Write sheets ───────────────────────────────────────────────────────
    SHEET_COLS = ["pair_id", "query", "restaurant_id", "sentence_text", "label", "note"]
    KEY_COLS   = ["pair_id", "query", "restaurant_id", "restaurant_rank",
                  "evidence_rank", "unit_score", "unit_id", "sentence_text"]

    def write_csv(path: Path, rows: list[dict], cols: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {len(rows):4d} rows → {path}")

    write_csv(CACHE / "b2_support_sheet.csv",      sheet_rows, SHEET_COLS)
    write_csv(CACHE / "b2_support_sheet_ann2.csv", ann2_pairs, SHEET_COLS)
    write_csv(CACHE / "b2_support_key.csv",        key_rows,   KEY_COLS)

    print("\nDone.")
    print("\nNext steps:")
    print("  1. Sam labels b2_support_sheet.csv (full sheet, 0/1/2 in 'label' column).")
    print("  2. Ton labels b2_support_sheet_ann2.csv WITHOUT seeing Sam's labels.")
    print("  3. Run: python baselines/b2_score_agreement.py  to compute kappa + % supporting.")


if __name__ == "__main__":
    main()
