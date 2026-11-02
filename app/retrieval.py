"""
retrieval.py — Query-time retrieval engine for the Food Recommendation demo.
Python 3.9 compatible.

Loads pre-computed evidence at startup (fast). Supports:
  - English: lookup from evidence_sentence.csv (100 known RIRD queries)
             + live inference for new queries via sentence-transformers
  - Thai: lookup from topk_n10000_s42.csv (24 frozen taste queries)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
CACHE  = ROOT / "baselines" / "cache"
THAI_R = ROOT / "thai" / "results"
DATA   = ROOT / "datasets" / "rir_data-main"

MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"
MAX_RESULTS = 10
K_EV        = 5   # evidence sentences to return per restaurant


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _agg_top_k_mean(scores: List[float], k: int = 10) -> float:
    top = sorted(scores, reverse=True)[:k]
    return float(np.mean(top)) if top else 0.0


# ── Main engine ────────────────────────────────────────────────────────────

class RetrievalEngine:
    """Loads all artefacts once at startup; answers queries in milliseconds."""

    def __init__(self):
        self.ready = False
        self._restaurant_names: Dict[str, str] = {}
        self._restaurant_categories: Dict[str, str] = {}

        # English pre-computed: query → ranked results
        self._en_precomputed: Dict[str, List[dict]] = {}
        self._en_queries: List[str] = []

        # English live inference
        self._embedder = None
        self._sentence_embs: Optional[np.ndarray] = None
        self._sentence_meta: List[dict] = []  # {restaurant_id, text, unit_id}

        # Thai pre-computed: query_th → ranked results
        self._th_precomputed: Dict[str, List[dict]] = {}
        self._th_queries: List[str] = []

    # ── Public ─────────────────────────────────────────────────────────────

    def load(self):
        logger.info("Loading retrieval engine...")
        self._load_restaurant_meta()
        self._load_english_evidence()
        self._load_thai_evidence()
        self._try_load_embeddings()
        self.ready = True
        logger.info(
            "Engine ready. EN queries: %d, TH queries: %d, Live inference: %s",
            len(self._en_queries),
            len(self._th_queries),
            "yes" if self._embedder else "no (pre-computed only)",
        )

    def query_english(self, query: str, n: int = MAX_RESULTS) -> dict:
        """Return ranked restaurants for an English query."""
        query = query.strip()
        matched = self._find_close_query(query, self._en_queries)

        if matched:
            results = self._en_precomputed[matched][:n]
            return {"results": results, "is_precomputed": True,
                    "matched_query": matched if matched != query else None}

        if self._embedder is not None:
            results = self._live_infer_english(query, n)
            return {"results": results, "is_precomputed": False, "matched_query": None}

        return {"results": [], "is_precomputed": False,
                "error": "Query not in pre-computed set and live inference not available.",
                "hint": "Try one of the suggested queries."}

    def query_thai(self, query: str, n: int = MAX_RESULTS) -> dict:
        """Return top sentences for a Thai taste query."""
        query = query.strip()
        matched = self._find_close_query(query, self._th_queries)

        if matched:
            results = self._th_precomputed[matched][:n]
            return {"results": results, "is_precomputed": True,
                    "matched_query": matched if matched != query else None}

        return {"results": [], "is_precomputed": False,
                "error": "Thai query not recognised. Please select from the suggested queries.",
                "hint": "Thai live inference is not supported in this demo."}

    def get_suggested_queries(self, lang: str) -> List[str]:
        if lang == "th":
            return self._th_queries
        return self._en_queries

    # ── Loaders ────────────────────────────────────────────────────────────

    def _load_restaurant_meta(self):
        rev_path = DATA / "reviews" / "50_restaurants_all_rates.csv"
        if not rev_path.exists():
            logger.warning("Reviews CSV not found: %s", rev_path)
            return
        seen = set()
        for row in _load_csv(rev_path):
            bid = row["business_id"]
            if bid not in seen:
                self._restaurant_names[bid] = row.get("name", bid)
                self._restaurant_categories[bid] = row.get("categories", "")
                seen.add(bid)
        logger.info("Loaded %d restaurant names", len(self._restaurant_names))

    def _load_english_evidence(self):
        """Load Dense | top-10 mean sentence-level evidence (pre-computed)."""
        ev_path = next(
            (p for p in [CACHE / "sentence" / "evidence_sentence.csv",
                          CACHE / "evidence_sentence.csv"] if p.exists()),
            None,
        )
        if ev_path is None:
            logger.warning("evidence_sentence.csv not found — English pre-computed unavailable")
            return

        raw: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        rank_by_query_rest: Dict[str, Dict[str, int]] = defaultdict(dict)
        score_by_query_rest: Dict[str, Dict[str, float]] = defaultdict(dict)

        for row in _load_csv(ev_path):
            if row["model"] != "Dense" or row["aggregation"] != "top-10 mean":
                continue
            q   = row["query"]
            rid = row["restaurant_id"]
            raw[q][rid].append((
                int(row["evidence_rank"]),
                float(row["unit_score"]),
                row["text"],
            ))
            rank_by_query_rest[q][rid]  = int(row["restaurant_rank"])
            score_by_query_rest[q][rid] = float(row["restaurant_score"])

        for q, rest_dict in raw.items():
            ranked_rests = sorted(
                rest_dict.keys(),
                key=lambda r: rank_by_query_rest[q].get(r, 999),
            )
            result_list = []
            for rid in ranked_rests:
                sentences = sorted(rest_dict[rid], key=lambda x: x[0])[:K_EV]
                result_list.append({
                    "rank":            rank_by_query_rest[q][rid],
                    "restaurant_id":   rid,
                    "restaurant_name": self._restaurant_names.get(rid, rid),
                    "categories":      self._restaurant_categories.get(rid, ""),
                    "score":           round(score_by_query_rest[q][rid], 4),
                    "explanation": [
                        {"text": t, "score": round(s, 4), "evidence_rank": er}
                        for er, s, t in sentences
                    ],
                })
            self._en_precomputed[q] = result_list

        self._en_queries = sorted(self._en_precomputed.keys())
        logger.info("Loaded %d English pre-computed queries", len(self._en_queries))

    def _load_thai_evidence(self):
        """Load Thai topk results (24 frozen queries, sentence level)."""
        th_path = THAI_R / "topk_n10000_s42.csv"
        if not th_path.exists():
            logger.warning("Thai results not found: %s", th_path)
            return

        raw: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        for row in _load_csv(th_path):
            q   = row.get("query_th", "").strip()
            sys = row.get("system", "dense")
            if not q:
                continue
            raw[q][sys].append((
                int(row.get("rank", 0)),
                float(row.get("score", 0)),
                row.get("text", ""),
                row.get("star_rating", ""),
            ))

        for q, sys_dict in raw.items():
            seen_texts: Dict[str, dict] = {}
            for sys_name, items in sys_dict.items():
                for rank, score, text, star in sorted(items, key=lambda x: x[0]):
                    key = text.strip()
                    if key not in seen_texts or score > seen_texts[key]["score"]:
                        seen_texts[key] = {
                            "text":   text,
                            "score":  round(score, 4),
                            "system": sys_name,
                            "star":   star,
                        }
            merged = sorted(seen_texts.values(), key=lambda x: -x["score"])
            self._th_precomputed[q] = merged

        self._th_queries = sorted(self._th_precomputed.keys())
        logger.info("Loaded %d Thai pre-computed queries", len(self._th_queries))

    def _try_load_embeddings(self):
        """Attempt to load sentence embeddings for live inference. Optional."""
        emb_path = CACHE / "sentence" / f"emb_{MODEL_NAME}_sentence_len256.npy"
        if not emb_path.exists():
            logger.info("Sentence embeddings not found — live inference disabled")
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(MODEL_NAME)
            logger.info("Loading sentence embeddings (~740 MB)...")
            self._sentence_embs = np.load(str(emb_path))

            ev_path = CACHE / "sentence" / "evidence_sentence.csv"
            if ev_path.exists():
                seen_ids: set = set()
                for row in _load_csv(ev_path):
                    uid = row.get("unit_id", "")
                    if uid and uid not in seen_ids:
                        self._sentence_meta.append({
                            "unit_id":       uid,
                            "restaurant_id": row["restaurant_id"],
                            "text":          row["text"],
                        })
                        seen_ids.add(uid)
            logger.info(
                "Live inference ready. Embeddings: %s, meta: %d",
                self._sentence_embs.shape,
                len(self._sentence_meta),
            )
        except Exception as e:
            logger.warning("Could not initialise live inference: %s", e)
            self._embedder = None

    def _live_infer_english(self, query: str, n: int) -> List[dict]:
        """Encode query, score all sentences, aggregate, rank restaurants."""
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = self._embedder.encode([query], convert_to_numpy=True)
        sims  = cosine_similarity(q_emb, self._sentence_embs)[0]

        rest_unit_scores: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
        for idx, meta in enumerate(self._sentence_meta):
            rest_unit_scores[meta["restaurant_id"]].append(
                (float(sims[idx]), meta["unit_id"], meta["text"])
            )

        rest_scores = {
            rid: _agg_top_k_mean([s for s, _, _ in units])
            for rid, units in rest_unit_scores.items()
        }
        ranked = sorted(rest_scores.keys(), key=lambda r: -rest_scores[r])

        results = []
        for rank, rid in enumerate(ranked[:n], start=1):
            units = sorted(rest_unit_scores[rid], key=lambda x: -x[0])[:K_EV]
            results.append({
                "rank":            rank,
                "restaurant_id":   rid,
                "restaurant_name": self._restaurant_names.get(rid, rid),
                "categories":      self._restaurant_categories.get(rid, ""),
                "score":           round(rest_scores[rid], 4),
                "explanation": [
                    {"text": t, "score": round(s, 4), "evidence_rank": i + 1}
                    for i, (s, _, t) in enumerate(units)
                ],
            })
        return results

    def _find_close_query(self, query: str, known: List[str]) -> Optional[str]:
        """Return exact match (case-insensitive), or None."""
        q = query.strip().lower()
        for k in known:
            if k.strip().lower() == q:
                return k
        return None
