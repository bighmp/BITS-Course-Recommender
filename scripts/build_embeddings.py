"""
Builds a semantic search index over course topics/descriptions using a
local embedding model (no API cost — fine for 500+ documents).

This is what lets a query like "AI-related DEL" or "lenient makeup policy"
match courses by meaning rather than exact keyword, before the LLM does
final explanation. Run this after extract_handouts.py.

Usage:
    python build_embeddings.py

Requires: sentence-transformers, numpy
    pip install sentence-transformers numpy
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

COURSES_PATH = Path("data/processed/courses.json")
INDEX_PATH = Path("data/embeddings/course_index.npy")
META_PATH = Path("data/embeddings/course_meta.json")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this scale


def course_text(course: dict) -> str:
    """What we embed: title + topics + handout topics + makeup/attendance
    text, since those are what natural-language queries reference."""
    parts = [
        course.get("title", ""),
        course.get("category") or "",
        " ".join(course.get("topics", [])),
    ]
    handout = course.get("handout") or {}
    parts.append(" ".join(handout.get("topics", [])))
    parts.append(handout.get("attendance_policy") or "")
    parts.append(handout.get("makeup_policy") or "")
    return " ".join(p for p in parts if p)


def main():
    courses = json.loads(COURSES_PATH.read_text())
    courses = [c for c in courses if "error" not in c]  # skip failed extractions

    model = SentenceTransformer(MODEL_NAME)
    texts = [course_text(c) for c in courses]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, embeddings)
    META_PATH.write_text(json.dumps([c["course_code"] for c in courses], indent=2))
    print(f"Indexed {len(courses)} courses -> {INDEX_PATH}")


_model_cache = None
_embeddings_cache = None
_course_codes_cache = None


def _get_resources():
    """Lazy-load once per process, then reuse. Without this, every single
    Streamlit query would reload the transformer model from disk — fine
    for a one-off CLI run, painfully slow for an interactive demo.

    IMPORTANT: all three must be assigned together, atomically, after
    every load succeeds. Assigning them one at a time meant that if
    np.load() failed (e.g. the embeddings file didn't exist yet), the
    model would already be cached while embeddings stayed None — and
    since the guard only checked "is the model cached", every later
    call would skip reloading and keep reusing that broken half-state
    for the rest of the process's life, even after the file was fixed.
    Streamlit doesn't restart its process on every rerun, so this kind
    of partial-failure caching bug persists silently across queries."""
    global _model_cache, _embeddings_cache, _course_codes_cache
    if _model_cache is None or _embeddings_cache is None or _course_codes_cache is None:
        model = SentenceTransformer(MODEL_NAME)
        embeddings = np.load(INDEX_PATH)
        course_codes = json.loads(META_PATH.read_text())
        _model_cache, _embeddings_cache, _course_codes_cache = model, embeddings, course_codes
    return _model_cache, _embeddings_cache, _course_codes_cache


def search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Cosine similarity search. Import this in requirement_engine.py /
    app.py rather than re-embedding on every query."""
    model, embeddings, course_codes = _get_resources()

    q_vec = model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_vec  # cosine sim, since both sides are normalized
    top_idx = np.argsort(-sims)[:top_k]
    return [(course_codes[i], float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    main()