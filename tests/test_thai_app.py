"""Tests for the Thai tab: precomputed lookup, BM25/dense merge, human labels, API schema."""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.retrieval import RetrievalEngine

SOUR = "อยากกินอะไรเปรี้ยวๆ"  # T01


@pytest.fixture(scope="module")
def engine():
    e = RetrievalEngine()
    e._load_thai_evidence()
    return e


@pytest.fixture(scope="module")
def client():
    # Skip the heavy model loading; the Thai tab only needs precomputed files.
    e = RetrievalEngine()
    e._load_thai_evidence()
    e.ready = True
    old_engine, old_lifespan = main.engine, main.app.router.lifespan_context
    main.engine = e
    main.app.router.lifespan_context = _no_lifespan
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        main.engine = old_engine
        main.app.router.lifespan_context = old_lifespan


@asynccontextmanager
async def _no_lifespan(app):
    yield


def test_all_24_queries_loaded(engine):
    assert len(engine.get_suggested_queries("th")) == 24


def test_merge_interleaves_systems_by_rank(engine):
    res = engine.query_thai(SOUR, n=20)["results"]
    # Both systems must appear near the top, not BM25 first and dense last.
    top4 = [set(r["systems"]) for r in res[:4]]
    assert any("bm25" in s for s in top4)
    assert any("dense" in s for s in top4)


def test_merge_dedupes_by_sentence(engine):
    res = engine.query_thai(SOUR, n=50)["results"]
    ids = [r["sent_id"] for r in res]
    assert len(ids) == len(set(ids))
    assert len(res) <= 20


def test_every_result_has_human_label(engine):
    # The 452-pair pool covers every top-10 sentence from both systems.
    for q in engine.get_suggested_queries("th"):
        for r in engine.query_thai(q, n=50)["results"]:
            assert r["label"] in (0, 1, 2), (q, r["sent_id"])


def test_known_label_value(engine):
    # P0001 in label_sheet.csv: T01, sent 32404_17, labelled 2, dense rank 3.
    res = engine.query_thai(SOUR, n=50)["results"]
    hit = next(r for r in res if r["sent_id"] == "32404_17")
    assert hit["label"] == 2
    assert hit["dense_rank"] == 3
    assert hit["systems"] == ["dense"]


def test_query_metadata(engine):
    out = engine.query_thai(SOUR)
    assert out["query_en"] == "I want something sour"


def test_api_thai_query_returns_200(client):
    r = client.post("/api/query", json={"query": SOUR, "lang": "th", "n_results": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"] == []
    assert len(body["sentences"]) == 10
    s = body["sentences"][0]
    assert {"text", "systems", "label", "star"} <= s.keys()


def test_api_unknown_thai_query(client):
    r = client.post("/api/query", json={"query": "ข้าวมันไก่", "lang": "th"})
    assert r.status_code == 200
    assert r.json()["sentences"] == []
    assert r.json()["error"]
