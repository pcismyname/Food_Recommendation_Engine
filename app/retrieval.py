"""
retrieval.py — Query-time retrieval engine for the Food Recommendation demo.
Python 3.9 compatible.

For pre-computed queries (100 English RIRD + 24 Thai):
  → instant lookup, no model needed.

For NEW English queries — selective B3 pipeline:
  Stage 0: LLM query rewriting  (Groq / qwen3.8-27b)
             Converts negation & vague phrasing to reviewer language.
  Stage 1: Bi-encoder retrieval (paraphrase-multilingual-mpnet-base-v2)
             Cosine similarity on rewritten query vs 240k sentence embeddings.
  Stage 2: Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
             Jointly re-scores (rewritten query, sentence) pairs per restaurant.
  Stage 3: Top-10 mean aggregation → restaurant ranking + top-K evidence.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
CACHE  = ROOT / "baselines" / "cache"
THAI   = ROOT / "thai"
THAI_R = THAI / "results"
DATA   = ROOT / "datasets" / "rir_data-main"

# ── Constants ─────────────────────────────────────────────────────────────────
BIENCODER_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
CE_MODEL          = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL        = "qwen/qwen3.8-27b"
MAX_RESULTS       = 10
MAX_LIVE_RESULTS  = 50    # matches the API's n_results upper bound
LIVE_CACHE_SIZE   = 256   # live queries kept in memory
K_AGG             = 10    # top-k sentence scores averaged per restaurant
K_EV              = 5     # evidence sentences returned per restaurant
CE_INPUT_K         = 15    # top-N bi-encoder candidates per restaurant sent to cross-encoder
CE_TOP_RESTS       = 20    # only cross-encode the top-N restaurants by bi-encoder score
MIN_SENTENCE_WORDS = 10    # filter out sentences shorter than this (removes "Solid meal!", etc.)
CE_DISPLAY_THRESH  = 0.0   # ms-marco CE threshold; fallback to top-2 if none pass

GROQ_SYSTEM_PROMPT = (
    "You are a query optimizer for a restaurant review search system. "
    "Rewrite the user's query so it matches the language that restaurant reviewers actually use.\n\n"
    "Rules:\n"
    "1. Convert negations to their POSITIVE OPPOSITE that a reviewer would write.\n"
    "   Example: \"no waitline\" → \"quick seating walk-in no reservation no queue\"\n"
    "   Example: \"not too noisy\" → \"quiet calm peaceful relaxed atmosphere\"\n"
    "2. Expand vague or idiomatic phrases to concrete food/restaurant attributes.\n"
    "   Example: \"cheat meal\" → \"indulgent rich comfort food burgers fries pizza calorie dense\"\n"
    "3. Keep the rewritten query SHORT (≤15 words), keyword-style, no punctuation.\n"
    "4. Return ONLY the rewritten query. No explanation. No preamble."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _agg_top_k_mean(scores: List[float], k: int = K_AGG) -> float:
    top = sorted(scores, reverse=True)[:k]
    return float(np.mean(top)) if top else 0.0


def _strip_think(text: str) -> str:
    """Remove Qwen3 chain-of-thought blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _split_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


# Negation/exclusion patterns that trigger cross-encoder re-ranking.
# For plain positive queries, bi-encoder alone beats ms-marco-MiniLM
# (which is trained on web search, not restaurant reviews).
_NEGATION_RE = re.compile(
    r"\b(no |not |without |don'?t |never |except |avoid |non[- ]|free of|minus |"
    r"less |instead of|rather than|other than|skip |exclude )\b",
    re.IGNORECASE,
)


# ── Main engine ───────────────────────────────────────────────────────────────

class RetrievalEngine:
    """Loads all artefacts once at startup and answers English queries live."""

    def __init__(self):
        self.ready = False

        # Restaurant metadata
        self._restaurant_names: Dict[str, str] = {}
        self._restaurant_categories: Dict[str, str] = {}

        # English pre-computed (100 RIRD queries → instant lookup)
        self._en_precomputed: Dict[str, List[dict]] = {}
        self._en_queries: List[str] = []

        # Thai pre-computed (24 frozen queries → instant lookup)
        self._th_precomputed: Dict[str, List[dict]] = {}
        self._th_queries: List[str] = []
        self._th_query_en: Dict[str, str] = {}

        # Stage 1: bi-encoder (lazy — only loaded if embeddings file exists)
        self._embedder = None
        self._sentence_embs: Optional[np.ndarray] = None
        self._sentence_meta: List[dict] = []
        self._sentence_unit_count = 0

        # Stage 2: cross-encoder (lazy — loaded alongside bi-encoder)
        self._cross_encoder = None

        # Stage 0: Groq client (lazy — loaded if GROQ_API_KEY available)
        self._groq_client = None
        self._rewrite_cache: Dict[str, str] = {}   # in-memory, cleared on restart
        # query → results, least recently used dropped first so memory stays bounded
        self._live_cache: "OrderedDict[str, dict]" = OrderedDict()

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self):
        logger.info("Loading retrieval engine...")
        self._load_restaurant_meta()
        self._load_english_evidence()
        self._load_thai_evidence()
        self._try_load_live_pipeline()
        self._try_init_groq()
        self.ready = True
        logger.info(
            "Engine ready | EN pre-computed: %d | TH pre-computed: %d | "
            "Live inference: %s | Query rewriting: %s",
            len(self._en_queries),
            len(self._th_queries),
            "yes (bi-encoder + cross-encoder)" if self._embedder else "no",
            "yes (Groq)" if self._groq_client else "no",
        )

    @property
    def capabilities(self) -> dict:
        return {
            "live_inference": self._embedder is not None,
            "query_rewriting": self._groq_client is not None,
            "cross_encoder": self._cross_encoder is not None,
        }

    def query_english(self, query: str, n: int = MAX_RESULTS) -> dict:
        """Return ranked restaurants for an English query."""
        query = query.strip()

        # 1. Try exact pre-computed lookup first (instant, highest quality)
        matched = self._find_close_query(query, self._en_queries)
        if matched:
            return {
                "results": self._en_precomputed[matched][:n],
                "is_precomputed": True,
                "matched_query": matched if matched.lower() != query.lower() else None,
                "rewritten_query": None,
                "pipeline": "pre-computed",
            }

        # 2. Live B3 pipeline: rewrite → bi-encode → cross-encode
        if self._embedder is not None:
            cache_key = query.lower()
            cached = self._live_cache.get(cache_key)
            if cached is not None:
                self._live_cache.move_to_end(cache_key)
                pipeline = cached["pipeline"] + " (cached)"
            else:
                rewritten = self._rewrite_query(query)
                # Always compute the maximum list so later requests with a larger n
                # can be served from the cache.
                results = self._live_infer_english(query, rewritten, MAX_LIVE_RESULTS)
                if self._cross_encoder is None:
                    pipeline = "rewrite+biencoder"
                elif self._needs_cross_encoder(query, rewritten):
                    pipeline = "rewrite+biencoder+crossencoder"
                else:
                    pipeline = "rewrite+biencoder (CE skipped: plain query)"
                cached = {"results": results, "rewritten_query": rewritten, "pipeline": pipeline}
                self._live_cache[cache_key] = cached
                if len(self._live_cache) > LIVE_CACHE_SIZE:
                    self._live_cache.popitem(last=False)
            rewritten = cached["rewritten_query"]
            return {
                "results": cached["results"][:n],
                "is_precomputed": False,
                "matched_query": None,
                "rewritten_query": rewritten if rewritten.lower() != query.lower() else None,
                "pipeline": pipeline,
            }

        return {
            "results": [],
            "is_precomputed": False,
            "rewritten_query": None,
            "error": "Query not in pre-computed set and live inference is not available.",
            "hint": "Try one of the suggested queries.",
        }

    def query_thai(self, query: str, n: int = MAX_RESULTS) -> dict:
        """Return top sentences for a Thai taste query (pre-computed only)."""
        query = query.strip()
        matched = self._find_close_query(query, self._th_queries)
        if matched:
            return {
                "results": self._th_precomputed[matched][:n],
                "query_en": self._th_query_en.get(matched),
                "is_precomputed": True,
                "matched_query": matched if matched.lower() != query.lower() else None,
                "rewritten_query": None,
            }
        return {
            "results": [],
            "is_precomputed": False,
            "rewritten_query": None,
            "error": "Thai query not recognised. Please select from the suggested queries.",
            "hint": "Thai live inference is not supported in this demo.",
        }

    def get_suggested_queries(self, lang: str) -> List[str]:
        return self._th_queries if lang == "th" else self._en_queries

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_restaurant_meta(self):
        rev_path = DATA / "reviews" / "50_restaurants_all_rates.csv"
        if not rev_path.exists():
            logger.warning("Reviews CSV not found: %s", rev_path)
            return
        seen: set = set()
        for row in _load_csv(rev_path):
            bid = row["business_id"]
            if bid not in seen:
                self._restaurant_names[bid]      = row.get("name", bid)
                self._restaurant_categories[bid] = row.get("categories", "")
                seen.add(bid)
        logger.info("Loaded %d restaurant names", len(self._restaurant_names))

    def _load_english_evidence(self):
        ev_path = next(
            (p for p in [CACHE / "sentence" / "evidence_sentence.csv",
                         CACHE / "evidence_sentence.csv"] if p.exists()),
            None,
        )
        if ev_path is None:
            logger.warning("evidence_sentence.csv not found — English pre-computed unavailable")
            return

        raw: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        rank_map: Dict[str, Dict[str, int]]   = defaultdict(dict)
        score_map: Dict[str, Dict[str, float]] = defaultdict(dict)

        for row in _load_csv(ev_path):
            if row["model"] != "Dense" or row["aggregation"] != "top-10 mean":
                continue
            q, rid = row["query"], row["restaurant_id"]
            raw[q][rid].append((int(row["evidence_rank"]), float(row["unit_score"]), row["text"]))
            rank_map[q][rid]  = int(row["restaurant_rank"])
            score_map[q][rid] = float(row["restaurant_score"])

        for q, rest_dict in raw.items():
            ranked = sorted(rest_dict.keys(), key=lambda r: rank_map[q].get(r, 999))
            self._en_precomputed[q] = [
                {
                    "rank":            rank_map[q][rid],
                    "restaurant_id":   rid,
                    "restaurant_name": self._restaurant_names.get(rid, rid),
                    "categories":      self._restaurant_categories.get(rid, ""),
                    "score":           round(score_map[q][rid], 4),
                    "explanation": [
                        {"text": t, "score": round(s, 4), "evidence_rank": er}
                        for er, s, t in sorted(rest_dict[rid], key=lambda x: x[0])[:K_EV]
                    ],
                }
                for rid in ranked
            ]

        self._en_queries = sorted(self._en_precomputed.keys())
        logger.info("Loaded %d English pre-computed queries", len(self._en_queries))

        # Reconstruct the full sentence corpus in the same order used to
        # generate the embedding matrix. Evidence CSV rows are only a pooled
        # subset, so using them positionally would pair the wrong text with
        # each embedding.
        rev_path = DATA / "reviews" / "50_restaurants_all_rates.csv"
        if rev_path.exists():
            unit_id = 0
            for row in _load_csv(rev_path):
                for text in _split_sentences(row.get("review_text", "")):
                    if len(text.split()) >= MIN_SENTENCE_WORDS:
                        self._sentence_meta.append({
                            "unit_id": unit_id,
                            "restaurant_id": row["business_id"],
                            "text": text,
                        })
                    unit_id += 1
            self._sentence_unit_count = unit_id
        else:
            logger.warning("Reviews CSV not found — live sentence metadata unavailable")
        logger.info("Sentence meta loaded: %d sentences (≥%d words)",
                    len(self._sentence_meta), MIN_SENTENCE_WORDS)

    def _load_thai_evidence(self):
        th_path = THAI_R / "topk_n10000_s42.csv"
        if not th_path.exists():
            logger.warning("Thai results not found: %s", th_path)
            return

        labels = self._load_thai_labels()

        qid_of: Dict[str, str] = {}
        ranked: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in _load_csv(th_path):
            q = row.get("query_th", "").strip()
            sys_name = row.get("system", "")
            if not q or sys_name not in ("bm25", "dense"):
                continue
            qid_of[q] = row.get("qid", "")
            ranked[q][sys_name].append(row)

        for q, by_sys in ranked.items():
            for rows in by_sys.values():
                rows.sort(key=lambda r: int(r["rank"]))
            self._th_precomputed[q] = self._merge_thai_systems(qid_of[q], by_sys, labels)

        self._th_queries = sorted(self._th_precomputed.keys())
        logger.info("Loaded %d Thai pre-computed queries", len(self._th_queries))

        q_path = THAI / "thai_queries.csv"
        if q_path.exists():
            self._th_query_en = {r["query_th"].strip(): r["query_en"] for r in _load_csv(q_path)}

    @staticmethod
    def _load_thai_labels() -> Dict[Tuple[str, str], int]:
        """Map (qid, sent_id) → human relevance label 0/1/2 from the C4 label sheet."""
        sheet, key = THAI / "labels" / "label_sheet.csv", THAI / "labels" / "pool_key.csv"
        if not (sheet.exists() and key.exists()):
            logger.warning("Thai labels not found — results will show without relevance badges")
            return {}
        label_of = {r["pair_id"]: r["label"].strip() for r in _load_csv(sheet)}
        out: Dict[Tuple[str, str], int] = {}
        for r in _load_csv(key):
            lab = label_of.get(r["pair_id"], "")
            if lab in ("0", "1", "2"):
                out[(r["qid"], r["sent_id"])] = int(lab)
        return out

    @staticmethod
    def _merge_thai_systems(qid: str, by_sys: Dict[str, List[dict]],
                            labels: Dict[Tuple[str, str], int]) -> List[dict]:
        """Interleave BM25 and dense by rank (BM25 first, as it scored higher in C7).

        Raw scores are not comparable (BM25 ~10–15, cosine < 1), so we merge by rank
        and keep each system's own rank and score on the result.
        """
        merged: Dict[str, dict] = {}
        order: List[str] = []
        systems = [s for s in ("bm25", "dense") if s in by_sys]
        depth = max((len(v) for v in by_sys.values()), default=0)
        for i in range(depth):
            for sys_name in systems:
                if i >= len(by_sys[sys_name]):
                    continue
                row = by_sys[sys_name][i]
                sid = row["sent_id"]
                if sid not in merged:
                    order.append(sid)
                    merged[sid] = {
                        "sent_id": sid,
                        "text": row["text"],
                        "star": row.get("star_rating", ""),
                        "systems": [],
                        "bm25_rank": None, "bm25_score": None,
                        "dense_rank": None, "dense_score": None,
                        "label": labels.get((qid, sid)),
                    }
                m = merged[sid]
                m["systems"].append(sys_name)
                m[f"{sys_name}_rank"] = int(row["rank"])
                m[f"{sys_name}_score"] = round(float(row["score"]), 4)
        return [merged[s] for s in order]

    def _try_load_live_pipeline(self):
        """Load bi-encoder + cross-encoder for live inference. Optional."""
        emb_path = CACHE / "sentence" / f"emb_{BIENCODER_MODEL}_sentence_len256.npy"
        if not emb_path.exists():
            logger.info("Sentence embeddings not found — live inference disabled")
            return
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder

            logger.info("Loading bi-encoder: %s", BIENCODER_MODEL)
            self._embedder = SentenceTransformer(BIENCODER_MODEL)

            logger.info("Loading sentence embeddings (~740 MB)...")
            self._sentence_embs = np.load(str(emb_path))

            logger.info("Loading cross-encoder: %s", CE_MODEL)
            self._cross_encoder = CrossEncoder(CE_MODEL)

            logger.info(
                "Live pipeline ready | embeddings: %s | meta: %d sentences",
                self._sentence_embs.shape, len(self._sentence_meta),
            )
            if self._sentence_embs.shape[0] != self._sentence_unit_count:
                raise ValueError(
                    "Sentence embedding and corpus sizes do not match: "
                    f"{self._sentence_embs.shape[0]} embeddings vs "
                    f"{self._sentence_unit_count} sentence units"
                )
        except Exception as e:
            logger.warning("Could not initialise live pipeline: %s", e)
            self._embedder       = None
            self._cross_encoder  = None

    def _try_init_groq(self):
        """Initialise Groq client for query rewriting. Optional."""
        groq_key = os.getenv("groq") or os.getenv("GROQ_API_KEY")
        if not groq_key:
            # Try loading .env manually
            env_path = ROOT / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
            groq_key = os.getenv("groq") or os.getenv("GROQ_API_KEY")

        if not groq_key:
            logger.info("No Groq API key found — query rewriting disabled")
            return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=groq_key)
            logger.info("Groq client initialised (model: %s)", GROQ_MODEL)
        except Exception as e:
            logger.warning("Could not initialise Groq client: %s", e)

    # ── Stage 0: Query rewriting ──────────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query using Groq LLM. Returns original if rewriting fails."""
        if query in self._rewrite_cache:
            return self._rewrite_cache[query]
        if self._groq_client is None:
            return query
        try:
            resp = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user",   "content": query + " /no_think"},
                ],
                temperature=0.1,
                max_tokens=60,
            )
            rewritten = _strip_think(resp.choices[0].message.content)
            self._rewrite_cache[query] = rewritten
            logger.info("Rewrote query: %r → %r", query, rewritten)
            return rewritten
        except Exception as e:
            logger.warning("Groq rewrite failed for %r: %s", query, e)
            return query

    # ── Stage 1 + 2 + 3: Live inference ──────────────────────────────────────

    def _needs_cross_encoder(self, query: str, rewritten: str) -> bool:
        """Return True only when the query contains negation/exclusion language.
        For plain positive queries, the bi-encoder alone gives better results
        because ms-marco-MiniLM-L-6-v2 is trained on web search, not restaurant reviews."""
        return bool(_NEGATION_RE.search(query) or _NEGATION_RE.search(rewritten))

    def _live_infer_english(self, original_query: str, rewritten_query: str, n: int) -> List[dict]:
        """
        Full B3 pipeline for a new English query:
          1. Bi-encoder cosine similarity on rewritten_query vs all sentence embeddings
          2. Cross-encoder re-score of top-k candidates per restaurant
          3. Top-10 mean aggregation → restaurant ranking
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Stage 1: Bi-encoder retrieval using rewritten query
        q_emb = self._embedder.encode([rewritten_query], convert_to_numpy=True)
        sims  = cosine_similarity(q_emb, self._sentence_embs)[0]

        # Group by restaurant
        rest_candidates: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for meta in self._sentence_meta:
            rest_candidates[meta["restaurant_id"]].append(
                (float(sims[meta["unit_id"]]), meta["text"])
            )

        # Keep top-20 bi-encoder candidates per restaurant for cross-encoder input
        # Only apply CE for negation/exclusion queries — for plain queries bi-encoder wins
        use_ce = self._cross_encoder is not None and self._needs_cross_encoder(original_query, rewritten_query)
        logger.info("Query %r: cross-encoder %s", original_query[:40], "ON" if use_ce else "OFF (bi-encoder only)")

        if use_ce:
            # Stage 2: Cross-encoder re-ranking
            # First select top-CE_TOP_RESTS restaurants by bi-encoder score (fast pre-filter)
            bi_scores = {
                rid: _agg_top_k_mean([s for s, _ in cands])
                for rid, cands in rest_candidates.items()
            }
            top_rests_bi = sorted(bi_scores.keys(), key=lambda r: -bi_scores[r])[:CE_TOP_RESTS]

            rest_ce_scores: Dict[str, float] = {}
            rest_top_sents: Dict[str, List[Tuple[float, str]]] = {}

            for rid in top_rests_bi:
                candidates = rest_candidates[rid]
                # Use top-CE_INPUT_K bi-encoder candidates, pre-filtered by length
                top_cands = [
                    (s, t) for s, t in sorted(candidates, key=lambda x: -x[0])[:CE_INPUT_K]
                    if len(t.split()) >= MIN_SENTENCE_WORDS
                ] or sorted(candidates, key=lambda x: -x[0])[:CE_INPUT_K]  # fallback if all filtered

                sentences = [t for _, t in top_cands]
                pairs     = [(rewritten_query, s) for s in sentences]
                ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False).tolist()
                scored    = sorted(zip(ce_scores, sentences), key=lambda x: -x[0])
                top_k     = scored[:K_AGG]
                rest_ce_scores[rid] = _agg_top_k_mean([s for s, _ in top_k], k=K_AGG)
                rest_top_sents[rid] = top_k

            # Stage 3: Rank restaurants by aggregated CE score
            ranked = sorted(rest_ce_scores.keys(), key=lambda r: -rest_ce_scores[r])
            results = []
            for rank_pos, rid in enumerate(ranked[:n], start=1):
                all_ev = rest_top_sents[rid]
                # Apply display threshold; always show at least 2 sentences
                above  = [(s, t) for s, t in all_ev if s >= CE_DISPLAY_THRESH]
                top_ev = (above if len(above) >= 2 else all_ev)[:K_EV]
                results.append({
                    "rank":            rank_pos,
                    "restaurant_id":   rid,
                    "restaurant_name": self._restaurant_names.get(rid, rid),
                    "categories":      self._restaurant_categories.get(rid, ""),
                    "score":           round(rest_ce_scores[rid], 4),
                    "explanation": [
                        {"text": t, "score": round(s, 4), "evidence_rank": i + 1}
                        for i, (s, t) in enumerate(top_ev)
                    ],
                })
        else:
            # Fallback: bi-encoder only (no cross-encoder)
            rest_bi_scores = {
                rid: _agg_top_k_mean([s for s, _ in cands])
                for rid, cands in rest_candidates.items()
            }
            ranked = sorted(rest_bi_scores.keys(), key=lambda r: -rest_bi_scores[r])
            results = []
            for rank_pos, rid in enumerate(ranked[:n], start=1):
                top_ev = sorted(rest_candidates[rid], key=lambda x: -x[0])[:K_EV]
                results.append({
                    "rank":            rank_pos,
                    "restaurant_id":   rid,
                    "restaurant_name": self._restaurant_names.get(rid, rid),
                    "categories":      self._restaurant_categories.get(rid, ""),
                    "score":           round(rest_bi_scores[rid], 4),
                    "explanation": [
                        {"text": t, "score": round(s, 4), "evidence_rank": i + 1}
                        for i, (s, t) in enumerate(top_ev)
                    ],
                })

        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def _find_close_query(self, query: str, known: List[str]) -> Optional[str]:
        q = query.strip().lower()
        for k in known:
            if k.strip().lower() == q:
                return k
        return None
