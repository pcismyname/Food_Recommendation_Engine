"""Tests for the English tab: precomputed lookup, live-path caching, API validation and schema."""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.retrieval as retrieval
from app.retrieval import RetrievalEngine


@pytest.fixture(scope="module")
def engine():
    # Precomputed data only; skip the bi-encoder, cross-encoder and Groq.
    e = RetrievalEngine()
    e._load_restaurant_meta()
    e._load_english_evidence()
    e.ready = True
    return e


@asynccontextmanager
async def _no_lifespan(app):
    yield


@pytest.fixture(scope="module")
def client(engine):
    old_engine, old_lifespan = main.engine, main.app.router.lifespan_context
    main.engine = engine
    main.app.router.lifespan_context = _no_lifespan
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        main.engine = old_engine
        main.app.router.lifespan_context = old_lifespan


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine_ready": True}


def test_all_100_queries_loaded(client):
    r = client.get("/api/queries", params={"lang": "en"})
    assert len(r.json()["queries"]) == 100


def test_precomputed_results_are_complete(engine):
    for q in engine.get_suggested_queries("en"):
        res = engine._en_precomputed[q]
        assert len(res) == 50, q                       # all RIRD restaurants ranked
        assert [r["rank"] for r in res] == list(range(1, 51)), q
        for r in res:
            assert r["restaurant_name"] != r["restaurant_id"]   # name was found
            assert 1 <= len(r["explanation"]) <= 5


def test_api_precomputed_query(client, engine):
    q = engine.get_suggested_queries("en")[0]
    r = client.post("/api/query", json={"query": q, "n_results": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_precomputed"] is True
    assert body["pipeline"] == "pre-computed"
    assert body["matched_query"] is None
    assert body["sentences"] == []
    assert len(body["results"]) == 10
    first = body["results"][0]
    assert {"rank", "restaurant_id", "restaurant_name", "categories", "score", "explanation"} <= first.keys()
    assert {"text", "score", "evidence_rank"} <= first["explanation"][0].keys()


def test_api_precomputed_is_case_and_space_insensitive(client, engine):
    q = engine.get_suggested_queries("en")[0]
    r = client.post("/api/query", json={"query": f"  {q.upper()}  "})
    assert r.json()["is_precomputed"] is True


def test_api_n_results_respected(client, engine):
    q = engine.get_suggested_queries("en")[0]
    r = client.post("/api/query", json={"query": q, "n_results": 3})
    assert len(r.json()["results"]) == 3


def test_api_unknown_query_without_live_model(client):
    r = client.post("/api/query", json={"query": "tacos for a picky toddler"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert body["error"] and body["hint"]


@pytest.mark.parametrize("body", [
    {"query": ""},
    {"query": "   "},
    {"query": "x" * 501},
    {"query": "pizza", "lang": "fr"},
    {"query": "pizza", "n_results": 0},
    {"query": "pizza", "n_results": 51},
])
def test_api_rejects_bad_input(client, body):
    assert client.post("/api/query", json=body).status_code == 422


def test_api_engine_not_ready_returns_503(client, engine):
    engine.ready = False
    try:
        assert client.post("/api/query", json={"query": "pizza"}).status_code == 503
    finally:
        engine.ready = True


# ── Live path (models replaced by a fake) ───────────────────────────────────

class _FakeLive(RetrievalEngine):
    """Live path with the model calls stubbed out, to test routing and caching."""

    def __init__(self):
        super().__init__()
        self._embedder = object()          # marks live inference as available
        self.calls = 0

    def _rewrite_query(self, query):
        return query + " rewritten"

    def _live_infer_english(self, original_query, rewritten_query, n):
        self.calls += 1
        return [{"rank": i, "restaurant_id": f"r{i}", "restaurant_name": f"R{i}",
                 "categories": "", "score": 1.0 / i, "explanation": []}
                for i in range(1, n + 1)]


def test_live_cache_respects_larger_n():
    e = _FakeLive()
    assert len(e.query_english("cozy place", n=5)["results"]) == 5
    out = e.query_english("cozy place", n=10)
    assert len(out["results"]) == 10
    assert out["pipeline"].endswith("(cached)")
    assert out["rewritten_query"] == "cozy place rewritten"
    assert e.calls == 1


def test_live_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(retrieval, "LIVE_CACHE_SIZE", 3)
    e = _FakeLive()
    for i in range(5):
        e.query_english(f"query {i}")
    assert list(e._live_cache) == ["query 2", "query 3", "query 4"]


def test_negation_routing():
    e = RetrievalEngine()
    assert e._needs_cross_encoder("somewhere quiet without a long wait", "")
    assert e._needs_cross_encoder("Not Thai but Asian", "")
    assert not e._needs_cross_encoder("cheat meal after the gym", "burgers fries pizza")
