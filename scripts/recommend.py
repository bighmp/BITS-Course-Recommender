"""
The natural-language query layer. Takes a free-text request like
"Suggest an AI-related DEL with no midsem" and turns it into ranked,
explained recommendations — using keyword rules + embedding similarity,
NOT an LLM. This means it works with zero API cost/quota risk; a small
LLM call can be layered on top later purely to make the explanation
prose nicer, without changing anything about how eligibility or
matching is decided.

Pipeline:
    query text
      -> parse_query()      : pull out explicit constraints (category,
                               no-midsem, no-attendance, etc.) via
                               keyword rules
      -> eligible courses    : from requirement_engine (already computed
                               from the student's profile)
      -> filter by category  : drop courses known to be the WRONG
                               category (e.g. a known-CDC course when
                               they asked for a DEL) — courses with no
                               known category pass through, since our
                               category map only covers CDC (see
                               extract_category_map.py's docstring)
      -> filter by handout attributes : has_midsem/has_compre/attendance/
                               makeup leniency, using known handout data;
                               unknown data is never treated as a match
                               OR a rejection — it's flagged "could not
                               be verified" per the spec's own instruction
      -> rank by embedding similarity to the free-text interest portion
      -> template-explain each result

Usage:
    from recommend import recommend
    results = recommend(profile, "suggest an AI-related DEL with no midsem", courses, rules, category_map)
"""

import json
import re
from pathlib import Path

from schema import Course, StudentProfile
from requirement_engine import compute_remaining_requirements, get_eligible_courses
from build_embeddings import search as embedding_search

CATEGORY_KEYWORDS = {
    "CDC": [r"\bcdc\b", r"core course"],
    "DEL": [r"\bdel\b", r"discipline elective"],
    "HUEL": [r"\bhuel\b", r"humanit(y|ies) elective"],
    "OPEL": [r"\bopel\b", r"open elective"],
}

NO_MIDSEM_RE = re.compile(r"no\s*mid-?\s*sem", re.IGNORECASE)
NO_COMPRE_RE = re.compile(r"no\s*compre", re.IGNORECASE)
NO_ATTENDANCE_RE = re.compile(r"no\s*attendance", re.IGNORECASE)
LENIENT_MAKEUP_RE = re.compile(r"lenient|flexible|easy(?:-going)?\s*makeup", re.IGNORECASE)
PROJECT_BASED_RE = re.compile(r"project[- ]based|project\s*evaluation", re.IGNORECASE)

# Phrases suggesting a makeup policy is genuinely strict — used as the
# "lenient" check's negative signal, since we have no numeric leniency
# score, just free text.
STRICT_MAKEUP_PHRASES = ["will not be granted", "no makeup", "not granted in any condition", "strictly"]


def parse_query(query: str) -> dict:
    """Rule-based, not an LLM — but covers every example phrasing in the
    spec ('AI-related DEL', 'no attendance requirement', 'lenient makeup
    policy', 'project-based evaluation')."""
    constraints = {
        "category": None,
        "no_midsem": bool(NO_MIDSEM_RE.search(query)),
        "no_compre": bool(NO_COMPRE_RE.search(query)),
        "no_attendance": bool(NO_ATTENDANCE_RE.search(query)),
        "lenient_makeup": bool(LENIENT_MAKEUP_RE.search(query)),
        "project_based": bool(PROJECT_BASED_RE.search(query)),
    }
    for category, patterns in CATEGORY_KEYWORDS.items():
        if any(re.search(p, query, re.IGNORECASE) for p in patterns):
            constraints["category"] = category
            break
    return constraints


def _attribute_check(course: Course, constraints: dict) -> tuple[bool, list[str]]:
    """Returns (passes, unverifiable_notes). A course is excluded only
    when we KNOW it fails a constraint — unknown data never excludes,
    it just gets flagged, per the spec: 'if a requested property is not
    explicitly available... the answer should state that it could not
    be verified.'"""
    h = course.handout
    notes = []

    if constraints["no_midsem"]:
        if h and h.has_midsem is True:
            return False, notes
        if not h or h.has_midsem is None:
            notes.append("midsem status could not be verified from the handout")

    if constraints["no_compre"]:
        if h and h.has_compre is True:
            return False, notes
        if not h or h.has_compre is None:
            notes.append("comprehensive exam status could not be verified from the handout")

    if constraints["no_attendance"]:
        if h and h.attendance_policy and "not linked" not in h.attendance_policy.lower():
            # has a stated attendance policy that isn't explicitly "not linked to evaluation"
            if "attend" in h.attendance_policy.lower():
                return False, notes
        if not h or not h.attendance_policy:
            notes.append("attendance policy could not be verified from the handout")

    if constraints["lenient_makeup"]:
        if h and h.makeup_policy:
            if any(p in h.makeup_policy.lower() for p in STRICT_MAKEUP_PHRASES):
                return False, notes
        else:
            notes.append("makeup policy could not be verified from the handout")

    if constraints["project_based"]:
        if h and h.evaluation_components:
            blob = " ".join(h.evaluation_components).lower()
            if "project" not in blob:
                return False, notes
        else:
            notes.append("evaluation components could not be verified from the handout")

    return True, notes


def _explain(course: Course, constraints: dict, notes: list[str], similarity: float, category_map: dict[str, str]) -> str:
    parts = []
    if constraints["category"]:
        known_category = category_map.get(course.course_code, course.category)
        if known_category == constraints["category"]:
            parts.append(f"confirmed {constraints['category']} — satisfies your remaining requirement")
        else:
            notes.append(f"category not confirmed in our data; treated as an eligible elective, not verified as {constraints['category']}")
    h = course.handout
    if constraints["no_midsem"] and h and h.has_midsem is False:
        parts.append("has no mid-semester exam")
    if constraints["no_attendance"] and h and h.attendance_policy:
        parts.append("attendance is not linked to evaluation")
    if constraints["lenient_makeup"] and h and h.makeup_policy:
        parts.append("makeup policy looks relatively flexible")
    if constraints["project_based"] and h and h.evaluation_components:
        parts.append("evaluation includes a project component")
    if not parts:
        parts.append(f"matches your query (topic similarity: {similarity:.2f})")
    explanation = "This course " + "; ".join(parts) + "."
    if notes:
        explanation += " Note: " + "; ".join(notes) + "."
    return explanation


def recommend(
    profile: StudentProfile,
    query: str,
    courses: list[Course],
    rules: list,
    category_map: dict[str, str],
    top_k: int = 5,
) -> list[dict]:
    constraints = parse_query(query)

    remaining = compute_remaining_requirements(profile, rules, category_map)
    eligible = get_eligible_courses(profile, courses, remaining, category_map)

    # category filter: exclude only courses we KNOW are the wrong category
    if constraints["category"]:
        eligible = [
            c for c in eligible
            if category_map.get(c.course_code, c.category) in (None, constraints["category"])
        ]

    # attribute filters
    filtered = []
    for c in eligible:
        passes, notes = _attribute_check(c, constraints)
        if passes:
            filtered.append((c, notes))

    if not filtered:
        return []

    # rank by embedding similarity to the raw query text
    ranked_codes = embedding_search(query, top_k=len(filtered) * 3)  # over-fetch, then intersect
    rank_lookup = {code: score for code, score in ranked_codes}
    filtered.sort(key=lambda pair: rank_lookup.get(pair[0].course_code, 0.0), reverse=True)

    # dedupe by course_code — multiple handout files for the same course
    # (different sections/years) shouldn't show up as separate results.
    # filtered is already sorted by rank at this point, so the first
    # occurrence kept per code is the highest-ranked one.
    seen_codes = set()
    deduped = []
    for course, notes in filtered:
        if course.course_code in seen_codes:
            continue
        seen_codes.add(course.course_code)
        deduped.append((course, notes))

    results = []
    for course, notes in deduped[:top_k]:
        similarity = rank_lookup.get(course.course_code, 0.0)
        results.append({
            "course_code": course.course_code,
            "title": course.title,
            "explanation": _explain(course, constraints, notes, similarity, category_map),
        })
    return results


if __name__ == "__main__":
    from requirement_engine import load_courses, load_rules, load_category_map

    courses = load_courses()
    rules = load_rules()
    category_map = load_category_map()

    test_profile = StudentProfile(
        campus="Pilani", admission_year=2023, degree="B.E. Computer Science",
        current_semester=5, completed_courses=["CS F211", "CS F212", "CS F213"],
        current_courses=[], interests=["AI", "security"],
    )

    for q in [
        "Suggest an AI-related DEL",
        "I want a course with no midsem",
        "Suggest courses with no attendance requirement and lenient makeup policy",
    ]:
        print(f"\n=== Query: {q!r} ===")
        for r in recommend(test_profile, q, courses, rules, category_map):
            print(f"- {r['course_code']} ({r['title']}): {r['explanation']}")