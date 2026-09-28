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
    get_course_category,
)
from build_embeddings import search as embedding_search


# ============================================================
# GEMINI
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

_gemini_client = None

if os.getenv("GEMINI_API_KEY"):
    try:
        _gemini_client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
    except Exception:
        _gemini_client = None


# ============================================================
# QUERY SCHEMA
# ============================================================

class QueryConstraints(BaseModel):
    category: str | None = Field(
        default=None,
        description="CDC, DEL, HUEL, OPEL, or null",
    )

    topics: list[str] = Field(
        default_factory=list,
        description="Academic topics/interests",
    )

    requires_no_attendance: bool = False
    requires_no_midsem: bool = False
    requires_no_compre: bool = False
    lenient_makeup: bool = False
    project_based: bool = False

    semantic_query: str = ""


# ============================================================
# LOCAL QUERY PARSER
# ============================================================

def extract_intent_local(query: str) -> QueryConstraints:
    """
    Deterministic fallback parser.

    No API calls.
    """

    q = query.lower()

    # ----------------------------
    # Category
    # ----------------------------

    category = None

    category_patterns = {
        "CDC": [
            r"\bcdcs?\b",
            r"discipline core",
            r"core course",
        ],
        "DEL": [
            r"\bdels?\b",
            r"discipline elective",
            r"discipline electives",
        ],
        "HUEL": [
            r"\bhuel(?:s)?\b",
            r"humanities elective",
            r"humanities electives",
        ],
        "OPEL": [
            r"\bopel(?:s)?\b",
            r"open elective",
            r"open electives",
        ],
    }

    for cat, patterns in category_patterns.items():
        if any(re.search(p, q) for p in patterns):
            category = cat
            break

    # ----------------------------
    # Constraints
    # ----------------------------

    requires_no_attendance = any(
        phrase in q
        for phrase in [
            "no attendance",
            "without attendance",
            "attendance not required",
            "no attendance requirement",
        ]
    )

    requires_no_midsem = any(
        phrase in q
        for phrase in [
            "no midsem",
            "no mid-sem",
            "no mid sem",
            "without midsem",
            "without mid-sem",
            "without mid sem",
            "no midsemester",
        ]
    )

    requires_no_compre = any(
        phrase in q
        for phrase in [
            "no compre",
            "no comprehensive",
            "without compre",
            "without comprehensive",
        ]
    )

    lenient_makeup = any(
        phrase in q
        for phrase in [
            "lenient makeup",
            "lenient make up",
            "flexible makeup",
            "flexible make up",
            "easy makeup",
            "easy-going makeup",
        ]
    )

    project_based = any(
        phrase in q
        for phrase in [
            "project based",
            "project-based",
            "project based evaluation",
            "has a project",
        ]
    )

    # ----------------------------
    # Academic topics
    # ----------------------------

    topic_keywords = {
        "artificial intelligence": [
            "ai",
            "artificial intelligence",
        ],
        "machine learning": [
            "machine learning",
            "ml",
        ],
        "deep learning": [
            "deep learning",
        ],
        "natural language processing": [
            "nlp",
            "natural language processing",
        ],
        "large language models": [
            "llm",
            "llms",
            "large language model",
            "large language models",
        ],
        "data science": [
            "data science",
        ],
        "data mining": [
            "data mining",
        ],
        "security": [
            "security",
            "cybersecurity",
            "cyber security",
        ],
        "cryptography": [
            "cryptography",
            "crypto",
        ],
        "computer networks": [
            "network",
            "networks",
            "computer networks",
        ],
        "operating systems": [
            "operating systems",
            "os",
        ],
        "computer architecture": [
            "computer architecture",
            "architecture",
        ],
        "programming languages": [
            "programming languages",
            "programming language",
        ],
        "algorithms": [
            "algorithms",
            "algorithm",
        ],
        "databases": [
            "database",
            "databases",
        ],
        "economics": [
            "economics",
            "economic",
        ],
        "finance": [
            "finance",
            "financial",
        ],
        "game theory": [
            "game theory",
        ],
        "mathematics": [
            "mathematics",
            "math",
        ],
        "language": [
        "language",
        "linguistics",
        "linguistic",
        ],
    }

    topics = []

    for topic, keywords in topic_keywords.items():
        if any(
            re.search(
                rf"\b{re.escape(keyword)}\b",
                q,
            )
            for keyword in keywords
        ):
            topics.append(topic)

    semantic_query = " ".join(topics)

    return QueryConstraints(
        category=category,
        topics=topics,
        requires_no_attendance=requires_no_attendance,
        requires_no_midsem=requires_no_midsem,
        requires_no_compre=requires_no_compre,
        lenient_makeup=lenient_makeup,
        project_based=project_based,
        semantic_query=semantic_query,
    )


# ============================================================
# GEMINI INTENT EXTRACTION
# ============================================================

@lru_cache(maxsize=128)
def extract_intent_llm(
    query: str,
    profile_degree: str,
) -> QueryConstraints:

    if _gemini_client is None:
        raise RuntimeError("Gemini API key is not configured")

    prompt = f"""
You are an intent-extraction component for a BITS academic
course recommender.

Student programme:
{profile_degree}

Extract the student's course-search preferences into JSON.

Rules:

1. Do NOT determine academic eligibility.
2. Do NOT invent BITS rules.
3. Only extract what the student asks for.
4. category must be CDC, DEL, HUEL, OPEL, or null.
5. requires_no_attendance is true only when explicitly requested.
6. requires_no_midsem is true only when explicitly requested.
7. requires_no_compre is true only when explicitly requested.
8. lenient_makeup is true for lenient/flexible makeup requests.
9. project_based is true for project-based evaluation requests.
10. topics should contain concise academic interests.
11. semantic_query should describe the student's interests.
12. Anything not mentioned should be null, false, or empty.

Student query:
"{query}"
"""

    response = _gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueryConstraints,
            temperature=0.1,
        ),
    )

    return QueryConstraints.model_validate_json(
        response.text
    )


# ============================================================
# UNIFIED INTENT EXTRACTION
# ============================================================

def extract_intent(
    query: str,
    profile_degree: str,
    use_gemini: bool = False,
) -> tuple[QueryConstraints, str]:

    if use_gemini:

        try:
            constraints = extract_intent_llm(
                query.strip(),
                profile_degree,
            )

            return constraints, "Gemini"

        except Exception as e:

            print(
                f"Gemini failed: {type(e).__name__}: {e}"
            )

            constraints = extract_intent_local(query)

            return (
                constraints,
                "Local fallback (Gemini failed)",
            )

    return (
        extract_intent_local(query),
        "Local parser",
    )


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

    h = course.handout
    notes = []

    # NO MIDSEM
    if constraints.requires_no_midsem:

        if h and h.has_midsem is True:
            return False, notes

        if not h or h.has_midsem is None:
            notes.append(
                "midsem status could not be verified"
            )

    # NO COMPRE
    if constraints.requires_no_compre:

        if h and h.has_compre is True:
            return False, notes

        if not h or h.has_compre is None:
            notes.append(
                "comprehensive exam status could not be verified"
            )

    # NO ATTENDANCE
    if constraints.requires_no_attendance:

        if h and h.attendance_policy:

            attendance = h.attendance_policy.lower()

            if (
                "not linked" not in attendance
                and "attend" in attendance
            ):
                return False, notes

        else:
            notes.append(
                "attendance policy could not be verified"
            )

    # LENIENT MAKEUP
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
                "makeup policy could not be verified"
            )

    # PROJECT BASED
    if constraints.project_based:

        if h and h.evaluation_components:

            blob = " ".join(
                h.evaluation_components
            ).lower()

            if "project" not in blob:
                return False, notes

        else:
            notes.append(
                "evaluation components could not be verified"
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
    category_map: dict,
    profile_degree: str,
) -> str:

    parts = []

    if constraints.category:

        known_category = get_course_category(
            category_map=category_map,
            programme=profile_degree,
            course_code=course.course_code,
            course_category=getattr(course, "category", None),
        )

        if known_category == constraints.category:
            parts.append(
                f"confirmed {constraints.category}"
            )

    h = course.handout

    if constraints.requires_no_midsem and h:
        if h.has_midsem is False:
            parts.append("has no mid-semester exam")

    if constraints.requires_no_attendance and h:
        if h.attendance_policy:
            if "not linked" in h.attendance_policy.lower():
                parts.append(
                    "attendance is not linked to evaluation"
                )

    if constraints.requires_no_compre and h:
        if h.has_compre is False:
            parts.append("has no comprehensive exam")

    if constraints.lenient_makeup and h:
        if h.makeup_policy:
            parts.append(
                "makeup policy appears relatively flexible"
            )

    if constraints.project_based and h:
        if h.evaluation_components:

            blob = " ".join(
                h.evaluation_components
            ).lower()

            if "project" in blob:
                parts.append(
                    "evaluation includes a project component"
                )

    if not parts:

        if constraints.topics:
            parts.append(
                "matches your interests in "
                + ", ".join(constraints.topics)
            )
        else:
            parts.append(
                f"matches your request based on semantic "
                f"similarity ({similarity:.2f})"
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
    use_gemini: bool = False,
) -> tuple[list[dict], str]:

    # --------------------------------------------------------
    # 1. UNDERSTAND QUERY
    # --------------------------------------------------------

    constraints, parser_used = extract_intent(
        query,
        profile.degree,
        use_gemini=use_gemini,
    )

    print("\n===== INTENT =====")
    print(f"Parser: {parser_used}")
    print(constraints.model_dump())
    print("==================\n")

    # --------------------------------------------------------
    # 2. REQUIREMENTS
    # --------------------------------------------------------

    remaining = compute_remaining_requirements(
        profile,
        rules,
        category_map,
    )

    # IMPORTANT: pass RULES, not remaining
    eligible = get_eligible_courses(
        profile,
        courses,
        rules,
        category_map,
    )

    # --------------------------------------------------------
    # 3. CATEGORY FILTER
    # --------------------------------------------------------

    if constraints.category:
        eligible = [
            c
            for c in eligible
            if get_course_category(
                category_map=category_map,
                programme=profile.degree,
                course_code=c.course_code,
                course_category=getattr(c, "category", None),
            ) == constraints.category
        ]

    # --------------------------------------------------------
    # 4. ATTRIBUTE FILTERS
    # --------------------------------------------------------

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
        return [], parser_used

    # --------------------------------------------------------
    # 5. SEMANTIC RANKING
    # --------------------------------------------------------

    profile_interests = ", ".join(profile.interests)

    semantic_query = (
        constraints.semantic_query.strip()
        if constraints.semantic_query.strip()
        else " ".join(constraints.topics)
    )

    # Explicit user query takes priority.
    # Profile interests are only used when the query
    # does not specify a meaningful topic.
    if not semantic_query:
        semantic_query = profile_interests

    if not semantic_query:
        semantic_query = query

    ranked_codes = embedding_search(
        semantic_query,
        top_k=max(
            len(filtered) * 3,
            top_k * 3,
        ),
    )

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

    # --------------------------------------------------------
    # 6. DEDUPLICATE
    # --------------------------------------------------------

    seen_codes = set()
    deduped = []

    for course, notes in filtered:

        code = course.course_code.strip().upper()

        if code in seen_codes:
            continue

        seen_codes.add(code)

        deduped.append(
            (course, notes)
        )

    # --------------------------------------------------------
    # 7. RESULTS
    # --------------------------------------------------------

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
                "category": get_course_category(
                    category_map=category_map,
                    programme=profile.degree,
                    course_code=course.course_code,
                    course_category=getattr(course, "category", None),
                ),
                "similarity": round(
                    similarity,
                    3,
                ),
                "explanation": _explain(
                    course,
                    constraints,
                    notes,
                    similarity,
                    category_map,
                    profile.degree,
                ),
            }
        )

    return results, parser_used