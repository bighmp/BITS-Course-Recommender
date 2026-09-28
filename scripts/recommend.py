import os
import re
from functools import lru_cache

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from schema import Course, StudentProfile
from requirement_engine import (
    compute_remaining_requirements,
    get_eligible_courses,
)
from build_embeddings import search as embedding_search


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# You said this is the model that works in your setup.
MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# LLM QUERY SCHEMA
# ============================================================

class QueryConstraints(BaseModel):
    category: str | None = Field(
        default=None,
        description="CDC, DEL, HUEL, OPEL, or null",
    )

    topics: list[str] = Field(
        default_factory=list,
        description="Academic topics/interests mentioned in the query",
    )

    requires_no_attendance: bool = Field(
        default=False,
        description="True only if the student explicitly wants no attendance requirement",
    )

    requires_no_midsem: bool = Field(
        default=False,
        description="True only if the student explicitly wants no mid-semester exam",
    )

    requires_no_compre: bool = Field(
        default=False,
        description="True only if the student explicitly wants no comprehensive exam",
    )

    lenient_makeup: bool = Field(
        default=False,
        description="True if the student prefers a lenient/flexible makeup policy",
    )

    project_based: bool = Field(
        default=False,
        description="True if the student prefers project-based evaluation",
    )

    semantic_query: str = Field(
        default="",
        description="Compact semantic search query for the student's interests",
    )


# ============================================================
# LLM INTENT EXTRACTION
# ============================================================

@lru_cache(maxsize=128)
def extract_intent_llm(
    query: str,
    profile_degree: str,
) -> QueryConstraints:
    """
    Uses Gemini to translate natural language into structured
    preferences.

    IMPORTANT:
    Gemini does NOT determine academic eligibility.
    """

    prompt = f"""
You are an intent-extraction component for a BITS academic
course recommender.

Student programme:
{profile_degree}

Extract the student's course-search preferences into JSON.

Rules:

1. Do NOT determine whether a course is academically eligible.
2. Do NOT invent BITS rules.
3. Only extract what the student is asking for.
4. category must be one of:
   CDC, DEL, HUEL, OPEL, or null.
5. requires_no_attendance is true only when the student explicitly
   asks for no attendance requirement.
6. requires_no_midsem is true only when the student explicitly
   asks for no midsem / no mid-semester exam.
7. requires_no_compre is true only when the student explicitly
   asks for no compre / comprehensive exam.
8. lenient_makeup is true when the student asks for lenient,
   flexible, easy-going, or similar makeup policy.
9. project_based is true when the student asks for project-based
   evaluation.
10. topics should contain concise academic interests.
11. semantic_query should be a concise description of the
    student's academic interests for semantic retrieval.
12. Anything not mentioned should be null, false, or empty.

Examples:

User:
"Suggest a DEL related to AI"

category:
DEL

topics:
["artificial intelligence", "machine learning"]

User:
"I want an OPEL with no attendance requirement"

category:
OPEL

requires_no_attendance:
true

User:
"Suggest an AI-related DEL with no midsem and preferably project based"

category:
DEL

topics:
["artificial intelligence", "machine learning"]

requires_no_midsem:
true

project_based:
true

Student query:
"{query}"
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueryConstraints,
            temperature=0.1,
        ),
    )

    return QueryConstraints.model_validate_json(response.text)


# ============================================================
# ATTRIBUTE FILTERING
# ============================================================

STRICT_MAKEUP_PHRASES = [
    "will not be granted",
    "no makeup",
    "not granted in any condition",
    "strictly",
]


def _attribute_check(
    course: Course,
    constraints: QueryConstraints,
) -> tuple[bool, list[str]]:
    """
    Returns:
        (passes, unverifiable_notes)

    A course is rejected only when we KNOW that it violates
    a requested constraint.

    Unknown data is NOT treated as a failure.
    """

    h = course.handout
    notes = []

    # --------------------------------------------------------
    # NO MIDSEM
    # --------------------------------------------------------

    if constraints.requires_no_midsem:
        if h and h.has_midsem is True:
            return False, notes

        if not h or h.has_midsem is None:
            notes.append(
                "midsem status could not be verified from the handout"
            )

    # --------------------------------------------------------
    # NO COMPRE
    # --------------------------------------------------------

    if constraints.requires_no_compre:
        if h and h.has_compre is True:
            return False, notes

        if not h or h.has_compre is None:
            notes.append(
                "comprehensive exam status could not be verified from the handout"
            )

    # --------------------------------------------------------
    # NO ATTENDANCE
    # --------------------------------------------------------

    if constraints.requires_no_attendance:
        if h and h.attendance_policy:
            attendance = h.attendance_policy.lower()

            # The existing project convention treats
            # "not linked" as satisfying the no-attendance request.
            if "not linked" not in attendance and "attend" in attendance:
                return False, notes

        else:
            notes.append(
                "attendance policy could not be verified from the handout"
            )

    # --------------------------------------------------------
    # LENIENT MAKEUP
    # --------------------------------------------------------

    if constraints.lenient_makeup:
        if h and h.makeup_policy:
            makeup = h.makeup_policy.lower()

            if any(
                phrase in makeup
                for phrase in STRICT_MAKEUP_PHRASES
            ):
                return False, notes
        else:
            notes.append(
                "makeup policy could not be verified from the handout"
            )

    # --------------------------------------------------------
    # PROJECT BASED
    # --------------------------------------------------------

    if constraints.project_based:
        if h and h.evaluation_components:
            blob = " ".join(
                h.evaluation_components
            ).lower()

            if "project" not in blob:
                return False, notes
        else:
            notes.append(
                "evaluation components could not be verified from the handout"
            )

    return True, notes


# ============================================================
# EXPLANATION
# ============================================================

def _explain(
    course: Course,
    constraints: QueryConstraints,
    notes: list[str],
    similarity: float,
    category_map: dict[str, str],
) -> str:
    parts = []

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if constraints.category:
        known_category = category_map.get(
            course.course_code,
            course.category,
        )

        if known_category == constraints.category:
            parts.append(
                f"confirmed {constraints.category} — satisfies your remaining requirement"
            )
        else:
            notes.append(
                f"category not confirmed in our data; "
                f"treated as an eligible elective, not verified as "
                f"{constraints.category}"
            )

    # --------------------------------------------------------
    # NO MIDSEM
    # --------------------------------------------------------

    h = course.handout

    if constraints.requires_no_midsem and h:
        if h.has_midsem is False:
            parts.append("has no mid-semester exam")

    # --------------------------------------------------------
    # NO ATTENDANCE
    # --------------------------------------------------------

    if constraints.requires_no_attendance and h:
        if h.attendance_policy:
            if "not linked" in h.attendance_policy.lower():
                parts.append("attendance is not linked to evaluation")

    # --------------------------------------------------------
    # NO COMPRE
    # --------------------------------------------------------

    if constraints.requires_no_compre and h:
        if h.has_compre is False:
            parts.append("has no comprehensive exam")

    # --------------------------------------------------------
    # MAKEUP
    # --------------------------------------------------------

    if constraints.lenient_makeup and h:
        if h.makeup_policy:
            parts.append("makeup policy appears relatively flexible")

    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    if constraints.project_based and h:
        if h.evaluation_components:
            blob = " ".join(
                h.evaluation_components
            ).lower()

            if "project" in blob:
                parts.append(
                    "evaluation includes a project component"
                )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    if not parts:
        parts.append(
            f"matches your request based on semantic similarity "
            f"({similarity:.2f})"
        )

    explanation = (
        "This course "
        + "; ".join(parts)
        + "."
    )

    if notes:
        explanation += (
            " Note: "
            + "; ".join(notes)
            + "."
        )

    return explanation


# ============================================================
# MAIN RECOMMENDER
# ============================================================

def recommend(
    profile: StudentProfile,
    query: str,
    courses: list[Course],
    rules: list,
    category_map: dict[str, str],
    top_k: int = 5,
) -> list[dict]:
    """
    Main recommendation pipeline.

    1. Gemini extracts natural-language intent.
    2. Requirement engine calculates remaining requirements.
    3. Requirement engine determines eligible courses.
    4. Hard constraints are applied deterministically.
    5. Embeddings rank the remaining courses.
    6. Results receive deterministic explanations.
    """

    # ========================================================
    # 1. LLM: UNDERSTAND USER QUERY
    # ========================================================

    constraints = extract_intent_llm(
        query.strip(),
        profile.degree,
    )

    # TEMPORARY DEBUG OUTPUT
    print("\n===== LLM INTENT =====")
    print(constraints.model_dump())
    print("======================\n")

    # ========================================================
    # 2. REQUIREMENT ENGINE
    # ========================================================

    remaining = compute_remaining_requirements(
        profile,
        rules,
        category_map,
    )

    eligible = get_eligible_courses(
        profile,
        courses,
        remaining,
        category_map,
    )

    # ========================================================
    # 3. CATEGORY FILTER
    # ========================================================

    if constraints.category:
        eligible = [
            c
            for c in eligible
            if category_map.get(
                c.course_code,
                c.category,
            ) in (
                None,
                constraints.category,
            )
        ]

    # ========================================================
    # 4. ATTRIBUTE FILTERS
    # ========================================================

    filtered = []

    for course in eligible:
        passes, notes = _attribute_check(
            course,
            constraints,
        )

        if passes:
            filtered.append(
                (course, notes)
            )

    if not filtered:
        return []

    # ========================================================
    # 5. SEMANTIC RANKING
    # ========================================================

    semantic_query = (
        constraints.semantic_query.strip()
        if constraints.semantic_query.strip()
        else " ".join(constraints.topics)
    )

    if not semantic_query:
        semantic_query = query

    ranked_codes = embedding_search(
        semantic_query,
        top_k=max(len(filtered) * 3, top_k * 3),
    )

    # embedding_search returns:
    # [(course_code, similarity), ...]

    rank_lookup = {
        code: score
        for code, score in ranked_codes
    }

    filtered.sort(
        key=lambda pair: rank_lookup.get(
            pair[0].course_code,
            0.0,
        ),
        reverse=True,
    )

    # ========================================================
    # 6. DEDUPLICATE
    # ========================================================

    seen_codes = set()
    deduped = []

    for course, notes in filtered:
        if course.course_code in seen_codes:
            continue

        seen_codes.add(course.course_code)
        deduped.append(
            (course, notes)
        )

    # ========================================================
    # 7. BUILD RESULTS
    # ========================================================

    results = []

    for course, notes in deduped[:top_k]:

        similarity = rank_lookup.get(
            course.course_code,
            0.0,
        )

        results.append(
            {
                "course_code": course.course_code,
                "title": course.title,
                "explanation": _explain(
                    course,
                    constraints,
                    notes,
                    similarity,
                    category_map,
                ),
            }
        )

    return results