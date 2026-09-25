"""
Builds a semantic search index over course topics/descriptions using a
local embedding model (no API cost — fine for 500+ documents).

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


def search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Cosine similarity search. Import this in requirement_engine.py /
    app.py rather than re-embedding on every query."""
    model = SentenceTransformer(MODEL_NAME)
    embeddings = np.load(INDEX_PATH)
    course_codes = json.loads(META_PATH.read_text())

    q_vec = model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_vec  # cosine sim, since both sides are normalized
    top_idx = np.argsort(-sims)[:top_k]
    return [(course_codes[i], float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    main()