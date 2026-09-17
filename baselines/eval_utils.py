"""
eval_utils.py — shared retrieval evaluation utilities
======================================================
All models produce rankings as:
    rankings: Dict[str, List[str]]
        key   = query string
        value = list of business_ids ordered best-first

Pass rankings + qrels into evaluate_model() to get a metrics dict.
Pass multiple model dicts into compare_models() for the cross-model table.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set


# ─────────────────────────────────────────────
# Per-query metric functions
# ─────────────────────────────────────────────

def average_precision(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean of precision@k values at each relevant hit."""
    if not relevant_ids:
        return 0.0
    hits, ap = 0, 0.0
    for rank, bid in enumerate(ranked_ids, start=1):
        if bid in relevant_ids:
            hits += 1
            ap += hits / rank
    return ap / len(relevant_ids)


def dcg_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int) -> float:
    """DCG@k using graded or binary relevance."""
    dcg = 0.0
    for rank, bid in enumerate(ranked_ids[:k], start=1):
        rel = relevance.get(bid, 0.0)
        dcg += rel / np.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int) -> float:
    """nDCG@k. relevance is a dict {business_id: score}."""
    ideal = sorted(relevance.values(), reverse=True)
    ideal_dcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal[:k]))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / ideal_dcg


def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for bid in ranked_ids[:k] if bid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for bid in ranked_ids[:k] if bid in relevant_ids)
    return hits / k


def mrr(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank (first relevant hit)."""
    for rank, bid in enumerate(ranked_ids, start=1):
        if bid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────
# Main evaluation function (one model, all queries)
# ─────────────────────────────────────────────

def evaluate_model(
    rankings: Dict[str, List[str]],
    binary_qrels: Dict[str, Set[str]],
    graded_qrels: Dict[str, Dict[str, float]] = None,
    ks: tuple = (5, 10),
) -> Dict[str, float]:
    """
    Evaluate one model across all queries.

    Parameters
    ----------
    rankings : {query -> [business_id, ...]} ranked best-first
    binary_qrels : {query -> set of relevant business_ids}
    graded_qrels : {query -> {business_id -> relevance_score}}
                   If None, binary relevance (0/1) is used for nDCG.
    ks : cut-off values for nDCG, Recall, Precision

    Returns
    -------
    Dict of metric_name -> mean value over queries
    """
    maps, mrrs = [], []
    ndcg     = {k: [] for k in ks}
    recall   = {k: [] for k in ks}
    precision = {k: [] for k in ks}

    for query, ranked in rankings.items():
        rel_set = binary_qrels.get(query, set())

        if graded_qrels and query in graded_qrels:
            rel_dict = graded_qrels[query]
        else:
            rel_dict = {bid: 1.0 for bid in rel_set}

        maps.append(average_precision(ranked, rel_set))
        mrrs.append(mrr(ranked, rel_set))

        for k in ks:
            ndcg[k].append(ndcg_at_k(ranked, rel_dict, k))
            recall[k].append(recall_at_k(ranked, rel_set, k))
            precision[k].append(precision_at_k(ranked, rel_set, k))

    results = {
        "MAP": np.mean(maps),
        "MRR": np.mean(mrrs),
    }
    for k in ks:
        results[f"nDCG@{k}"]   = np.mean(ndcg[k])
        results[f"Recall@{k}"] = np.mean(recall[k])
        results[f"P@{k}"]      = np.mean(precision[k])

    return results


# ─────────────────────────────────────────────
# Cross-model comparison table
# ─────────────────────────────────────────────

def compare_models(model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a tidy DataFrame comparing all models side by side.

    Parameters
    ----------
    model_results : {model_name -> metrics_dict}
        e.g. {"BM25": {...}, "Dense (no fusion)": {...}, "Top-3 Mean": {...}}

    Returns
    -------
    pd.DataFrame — rows = models, columns = metrics
    """
    df = pd.DataFrame(model_results).T
    df.index.name = "Model"

    col_order = ["MAP", "MRR"]
    for k in [5, 10]:
        for m in ["nDCG", "Recall", "P"]:
            col = f"{m}@{k}"
            if col in df.columns:
                col_order.append(col)
    df = df[[c for c in col_order if c in df.columns]]

    return df.round(4)
