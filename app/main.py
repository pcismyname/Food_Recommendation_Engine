"""
main.py — FastAPI application for the Food Recommendation Engine demo.
Python 3.9 compatible.

Endpoints
---------
GET  /                      → serve index.html
GET  /api/queries?lang=en   → list of known queries (for autocomplete)
POST /api/query             → run a query, return ranked results
GET  /api/health            → liveness check

Run locally:
    uvicorn app.main:app --reload --port 8000

Run in Docker:
    docker compose up
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Global engine ────────────────────────────────────────────────────────────

engine = RetrievalEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Explainable Food Recommendation Engine",
    description=(
        "Retrieval-based restaurant recommender with sentence-level explanations. "
        "English evaluation on RIRD; Thai taste case study on Wongnai."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    lang: Literal["en", "th"] = "en"
    n_results: int = Field(default=10, ge=1, le=50)


class ExplanationSentence(BaseModel):
    text: str
    score: float
    evidence_rank: int


class RestaurantResult(BaseModel):
    rank: int
    restaurant_id: str
    restaurant_name: str
    categories: str
    score: float
    explanation: list[ExplanationSentence]


class QueryResponse(BaseModel):
    query: str
    lang: str
    is_precomputed: bool
    matched_query: Optional[str]
    results: List[RestaurantResult]
    error: Optional[str] = None
    hint: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_ready": engine.ready}


@app.get("/api/queries")
def get_queries(lang: str = "en"):
    return {"queries": engine.get_suggested_queries(lang)}


@app.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Engine not ready yet. Try again shortly.")

    if req.lang == "en":
        raw = engine.query_english(req.query, req.n_results)
    else:
        raw = engine.query_thai(req.query, req.n_results)

    return QueryResponse(
        query=req.query,
        lang=req.lang,
        is_precomputed=raw.get("is_precomputed", False),
        matched_query=raw.get("matched_query"),
        results=[RestaurantResult(**r) for r in raw.get("results", [])],
        error=raw.get("error"),
        hint=raw.get("hint"),
    )
