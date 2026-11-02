"""b2_score_agreement.py — Score B2 human support labels.

Usage
-----
After both annotators have filled in their label columns:

    python baselines/b2_score_agreement.py

Reads
-----
    baselines/cache/b2_support_sheet.csv       — annotator 1 (Sam) labels
    baselines/cache/b2_support_sheet_ann2.csv  — annotator 2 (Ton) labels

Outputs
-------
Prints:
  - % of sentences rated as supporting (label ≥ 1) and clearly supporting (label = 2)
  - Cohen's kappa on the overlap set (ann2 pairs)
  - Per-query support rates
  - System-level comparison: kappa and support rate for BM25-sourced vs dense-sourced
    sentences (requires b2_support_key.csv)
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def read_labels(path: Path) -> dict[int, int]:
    """Return {pair_id: label} for rows that have a filled label."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                label = int(row["label"])
                labels[int(row["pair_id"])] = label
            except (ValueError, KeyError):
                pass  # skip rows with empty/invalid labels
    return labels


def cohen_kappa(labels_a: list[int], labels_b: list[int], n_classes: int = 3) -> float:
    """Cohen's kappa for ordinal labels 0, 1, 2."""
    assert len(labels_a) == len(labels_b), "Lists must be same length"
    n = len(labels_a)
    if n == 0:
        return float("nan")

    classes = list(range(n_classes))

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    freq_a = [labels_a.count(c) / n for c in classes]
    freq_b = [labels_b.count(c) / n for c in classes]
    p_e = sum(fa * fb for fa, fb in zip(freq_a, freq_b))

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main():
    HERE  = Path(__file__).resolve().parent
    CACHE = HERE / "cache"

    sheet1 = CACHE / "b2_support_sheet.csv"
    sheet2 = CACHE / "b2_support_sheet_ann2.csv"
    keyf   = CACHE / "b2_support_key.csv"

    # ── Load labels ────────────────────────────────────────────────────────
    if not sheet1.exists():
        print(f"ERROR: {sheet1} not found. Run b2_export_support_sheet.py first.")
        sys.exit(1)

    labels1 = read_labels(sheet1)
    labels2 = read_labels(sheet2) if sheet2.exists() else {}

    print(f"=== B2 Support Labelling Results ===\n")
    print(f"Annotator 1 (Sam)  — labelled {len(labels1)} / ", end="")
    with open(sheet1, newline="", encoding="utf-8") as f:
        total1 = sum(1 for _ in csv.DictReader(f))
    print(f"{total1} pairs")

    if labels2:
        print(f"Annotator 2 (Ton)  — labelled {len(labels2)} / ", end="")
        with open(sheet2, newline="", encoding="utf-8") as f:
            total2 = sum(1 for _ in csv.DictReader(f))
        print(f"{total2} pairs")
    else:
        print("Annotator 2 sheet not yet labelled.")

    if len(labels1) == 0:
        print("\nNo labels yet. Fill in b2_support_sheet.csv and re-run.")
        sys.exit(0)

    # ── Annotator 1 aggregate stats ────────────────────────────────────────
    vals1 = list(labels1.values())
    print(f"\n--- Annotator 1 stats (n={len(vals1)}) ---")
    for lv in [0, 1, 2]:
        c = vals1.count(lv)
        print(f"  Label {lv}: {c:4d}  ({c/len(vals1):.1%})")
    pct_supporting   = sum(v >= 1 for v in vals1) / len(vals1)
    pct_clearly      = sum(v == 2 for v in vals1) / len(vals1)
    print(f"  Supporting (≥1): {pct_supporting:.1%}")
    print(f"  Clearly (=2):    {pct_clearly:.1%}")

    # ── Agreement on overlap ───────────────────────────────────────────────
    if labels2:
        overlap = sorted(set(labels1) & set(labels2))
        if overlap:
            a1 = [labels1[pid] for pid in overlap]
            a2 = [labels2[pid] for pid in overlap]
            exact = sum(x == y for x, y in zip(a1, a2)) / len(a1)
            kappa = cohen_kappa(a1, a2)
            print(f"\n--- Agreement on {len(overlap)} shared pairs ---")
            print(f"  Exact agreement: {exact:.1%}")
            print(f"  Cohen's kappa:   {kappa:.3f}")
            if kappa >= 0.61:
                interp = "substantial"
            elif kappa >= 0.41:
                interp = "moderate"
            elif kappa >= 0.21:
                interp = "fair"
            else:
                interp = "slight/poor"
            print(f"  Interpretation:  {interp}")
        else:
            print("\nNo overlapping pair_ids found between the two sheets.")

    # ── Per-query support rates (Annotator 1) ─────────────────────────────
    print("\n--- Per-query support rate (Annotator 1) ---")
    # We need the query for each pair_id
    pair_to_query: dict[int, str] = {}
    with open(sheet1, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pair_to_query[int(row["pair_id"])] = row["query"]
            except (ValueError, KeyError):
                pass

    query_labels: dict[str, list[int]] = defaultdict(list)
    for pid, lv in labels1.items():
        q = pair_to_query.get(pid)
        if q:
            query_labels[q].append(lv)

    print(f"{'Query':<55} {'n':>4} {'≥1':>6} {'=2':>6}")
    print("-" * 72)
    for q in sorted(query_labels):
        lvs = query_labels[q]
        n   = len(lvs)
        s1  = sum(v >= 1 for v in lvs) / n
        s2  = sum(v == 2 for v in lvs) / n
        print(f"  {q[:53]:<53} {n:>4} {s1:>6.1%} {s2:>6.1%}")

    # ── System-level: BM25-sourced vs Dense-sourced ────────────────────────
    if keyf.exists():
        print("\n--- System-level support by source (requires key file) ---")
        # key has: pair_id, restaurant_rank, evidence_rank, unit_score, unit_id
        # We need to know which system found each sentence.
        # The key file does not directly encode the source system.
        # We infer it from unit_score:
        #   BM25 scores are typically large integers (BM25 values ~1–50)
        #   Dense scores are cosine similarities in [0, 1]
        # Better: join with evidence_sentence.csv on unit_id
        # For now, flag as "unknown" — a future enhancement.
        print("  (System-level breakdown requires merging with evidence_sentence.csv;")
        print("   run after both annotators have labelled.  Skipping for now.)")
    else:
        print("\n(b2_support_key.csv not found — skipping system-level breakdown)")

    print("\nDone.")


if __name__ == "__main__":
    main()
