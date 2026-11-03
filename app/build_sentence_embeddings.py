"""
Build the sentence-embedding matrix that turns on live inference in the demo.

The app only runs the live pipeline (Groq rewrite -> bi-encoder -> cross-encoder)
when this file exists:
    baselines/cache/sentence/emb_paraphrase-multilingual-mpnet-base-v2_sentence_len256.npy

It is gitignored (~740 MB), so build it once per machine:
    python -m app.build_sentence_embeddings

Sentences are produced with the app's own loader and splitter, so row i of the
matrix is guaranteed to line up with unit_id i in RetrievalEngine.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from app.retrieval import BIENCODER_MODEL, CACHE, DATA, _load_csv, _split_sentences

MAX_SEQ_LEN = 256  # matches baselines/baseline_models_v2_sentence.ipynb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def collect_sentences() -> list[str]:
    rev_path = DATA / "reviews" / "50_restaurants_all_rates.csv"
    sentences: list[str] = []
    for row in _load_csv(rev_path):
        sentences.extend(_split_sentences(row.get("review_text", "")))
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = parser.parse_args()

    out_path = CACHE / "sentence" / f"emb_{BIENCODER_MODEL}_sentence_len{MAX_SEQ_LEN}.npy"
    sentences = collect_sentences()
    logger.info("Collected %d sentences", len(sentences))

    if out_path.exists() and not args.force:
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] == len(sentences):
            logger.info("Already up to date: %s %s", out_path, existing.shape)
            return
        logger.info("Existing file has %d rows, expected %d - rebuilding",
                    existing.shape[0], len(sentences))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BIENCODER_MODEL)
    model.max_seq_length = MAX_SEQ_LEN
    logger.info("Encoding on %s", model.device)
    embs = model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embs)
    logger.info("Saved %s %s", out_path, embs.shape)


if __name__ == "__main__":
    main()
