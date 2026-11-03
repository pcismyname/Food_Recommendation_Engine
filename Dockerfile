# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Group Flower – AIT DSAI 2026"
LABEL description="Explainable Food Recommendation Engine demo"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/model_cache \
    HF_HOME=/model_cache

WORKDIR /app_root

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY app/ ./app/

# Copy data artefacts needed at runtime
# (datasets and cache are mounted as volumes in docker-compose for local dev,
#  but baked in for EC2 if you run `docker build` after downloading them)
COPY baselines/cache/sentence/evidence_sentence.csv  ./baselines/cache/sentence/evidence_sentence.csv
COPY thai/results/topk_n10000_s42.csv                ./thai/results/topk_n10000_s42.csv
COPY thai/thai_queries.csv                           ./thai/thai_queries.csv
COPY thai/labels/label_sheet.csv                     ./thai/labels/label_sheet.csv
COPY thai/labels/pool_key.csv                        ./thai/labels/pool_key.csv
COPY datasets/rir_data-main/reviews/50_restaurants_all_rates.csv \
                                                     ./datasets/rir_data-main/reviews/50_restaurants_all_rates.csv

# Optional: bake sentence embeddings in if you have the disk space (~750 MB)
# COPY baselines/cache/sentence/emb_paraphrase-multilingual-mpnet-base-v2_sentence_len256.npy \
#      ./baselines/cache/sentence/emb_paraphrase-multilingual-mpnet-base-v2_sentence_len256.npy

# Create model cache directory
RUN mkdir -p /model_cache

EXPOSE 8000

# Health check (waits 30s for model to load then checks every 15s)
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
