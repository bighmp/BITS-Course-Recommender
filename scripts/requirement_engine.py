"""
Requirement engine for the BITS Academic Course Recommender.

Responsibilities:
1. Load processed course/rule/category data.
2. Normalize programme names.
3. Determine the category of a course for a specific programme.
4. Compute completed / remaining requirements.
5. Check prerequisite satisfaction.
6. Produce a deterministic set of eligible courses.

Important:
- Programme-specific CDC/DEL mappings are authoritative.
- Global HUEL mappings are used for Humanities Electives.
- Unknown categories are NOT treated as eligible.
- Requirements with min_required=None are NOT assumed to be zero.
  They are reported as "unverified".
- The engine does not invent academic rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COURSES_PATH = PROCESSED_DIR / "courses.json"
RULES_PATH = PROCESSED_DIR / "programme_rules.json"
CATEGORY_MAP_PATH = PROCESSED_DIR / "course_category_map.json"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from schema import Course, ProgrammeRule, StudentProfile


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_programme(programme: str) -> str:
    """
    Normalize programme names so formatting differences in the bulletin
    do not affect matching.
    """
    if not programme:
        return ""

    value = str(programme).strip().lower()

    # Remove spaces around periods first:
    # "B. E." -> "B.E."
    value = re.sub(r"\s*\.\s*", ".", value)

    # Remove periods from degree abbreviations:
    # "B.E." -> "BE"
    value = value.replace(".", "")

    # Normalize separators.
    value = value.replace("_", " ")
    value = value.replace("-", " ")

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_code(course_code: str) -> str:
    """
    Normalize a course code.

    Example:
        "CS F407" -> "CS F407"
        "cs-f407" -> "CS F407"
        "CSF407"  -> "CS F407"
    """
    if not course_code:
        return ""

    value = str(course_code).strip().upper()

    # Normalize separators.
    value = value.replace("-", " ")
    value = value.replace("_", " ")

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value)

    # Handle codes accidentally written without a space:
    # CSF407 -> CS F407
    match = re.fullmatch(r"([A-Z]{2,6})\s*([A-Z]?\d{3,4})", value)

    if match:
        return f"{match.group(1)} {match.group(2)}"

    return value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_courses() -> List[Course]:
    """
    Load processed courses.

    The extraction pipeline may preserve records describing extraction
    failures, e.g.:

        {"source_file": "...", "error": "course_code not found"}

    These are not actual Course objects, so they are skipped rather than
    causing the entire requirement engine to fail.
    """
    raw = load_json(COURSES_PATH)

    if isinstance(raw, dict):
        if "courses" in raw:
            raw = raw["courses"]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected courses.json to contain a list, "
            f"got {type(raw).__name__}"
        )

    courses: List[Course] = []
    skipped = []

    for index, item in enumerate(raw):
        try:
            course = Course.model_validate(item)
            courses.append(course)

        except ValidationError as exc:
            skipped.append({
                "index": index,
                "source_file": item.get("source_file", "unknown"),
                "error": item.get("error", str(exc)),
            })

    print(f"Valid courses: {len(courses)}")
    print(f"Skipped invalid extraction records: {len(skipped)}")

    if skipped:
        print("\nFirst skipped records:")

        for record in skipped[:10]:
            print(
                f" - [{record['index']}] "
                f"{record['source_file']}: "
                f"{record['error']}"
            )

        if len(skipped) > 10:
            print(
                f" ... and {len(skipped) - 10} more"
            )

    return courses


def load_rules() -> List[ProgrammeRule]:
    """Load programme requirement rules."""
    raw = load_json(RULES_PATH)

    if isinstance(raw, dict):
        if "rules" in raw:
            raw = raw["rules"]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected programme_rules.json to contain a list, "
            f"got {type(raw).__name__}"
        )

    return [ProgrammeRule.model_validate(item) for item in raw]


def load_category_map() -> Dict[str, Any]:
    """Load the programme-aware course category map."""
    raw = load_json(CATEGORY_MAP_PATH)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected course_category_map.json to contain an object, "
            f"got {type(raw).__name__}"
        )

    return raw


# ---------------------------------------------------------------------------
# Programme category lookup
# ---------------------------------------------------------------------------

def get_programme_categories(
    category_map: Dict[str, Any],
    programme: str,
) -> Dict[str, List[str]]:
    """
    Return the authoritative CDC/DEL mapping for a programme.

    Expected structure:

    {
        "__PROGRAMME_CATEGORIES__": {
            "B.E. Computer Science": {
                "CDC": [...],
                "DEL": [...]
            }
        }
    }

    The extractor already handles the difficult PDF parsing.
    This function only performs normalized lookup.
    """

    programmes = category_map.get("__PROGRAMME_CATEGORIES__", {})

    if not isinstance(programmes, dict):
        return {}

    target = normalize_programme(programme)

    for programme_name, categories in programmes.items():
        if normalize_programme(programme_name) == target:
            if isinstance(categories, dict):
                return categories
            return {}

    return {}


def get_course_category(
    category_map: Dict[str, Any],
    programme: str,
    course_code: str,
    course_category: Optional[str] = None,
) -> Optional[str]:
    """
    Determine a course's category for a specific programme.

    Priority:

    1. Programme-specific CDC
    2. Programme-specific DEL
    3. Global HUEL
    4. Explicit category already attached to Course
    5. None / unknown

    We deliberately do NOT infer OPEL from an unknown course.
    """

    code = normalize_code(course_code)

    programme_categories = get_programme_categories(
        category_map,
        programme,
    )

    # ---------------------------------------------------------------
    # Programme-specific CDC
    # ---------------------------------------------------------------

    cdc_courses = programme_categories.get("CDC", [])

    if isinstance(cdc_courses, list):
        normalized_cdc = {
            normalize_code(item)
            for item in cdc_courses
        }

        if code in normalized_cdc:
            return "CDC"

    # ---------------------------------------------------------------
    # Programme-specific DEL
    # ---------------------------------------------------------------

    del_courses = programme_categories.get("DEL", [])

    if isinstance(del_courses, list):
        normalized_del = {
            normalize_code(item)
            for item in del_courses
        }

        if code in normalized_del:
            return "DEL"

    # ---------------------------------------------------------------
    # Global HUEL
    # ---------------------------------------------------------------

    global_categories = category_map.get(
        "__GLOBAL_CATEGORIES__",
        {},
    )

    if isinstance(global_categories, dict):
        huel_courses = global_categories.get("HUEL", [])

        if isinstance(huel_courses, list):
            normalized_huel = {
                normalize_code(item)
                for item in huel_courses
            }

            if code in normalized_huel:
                return "HUEL"

    # ---------------------------------------------------------------
    # Explicit course-level category
    # ---------------------------------------------------------------

    if course_category:
        category = str(course_category).strip().upper()

        if category in {"CDC", "DEL", "HUEL", "OPEL"}:
            return category

    return None


# ---------------------------------------------------------------------------
# Requirement computation
# ---------------------------------------------------------------------------

def compute_remaining_requirements(
    profile: StudentProfile,
    rules: List[ProgrammeRule],
    category_map: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute completed and remaining requirements.

    Rules with a numeric min_required are considered verified.

    Rules with min_required=None are reported as:

        status = "unverified"
        required = None
        remaining = None

    We intentionally do not convert None -> 0.
    """

    programme = profile.degree

    programme_rules = [
        rule
        for rule in rules
        if normalize_programme(rule.programme)
        == normalize_programme(programme)
    ]

    # Count completed courses by category.
    completed_counts = {
        "CDC": 0,
        "DEL": 0,
        "HUEL": 0,
        "OPEL": 0,
    }

    for completed_code in profile.completed_courses:
        code = normalize_code(completed_code)

        # We don't necessarily have a Course object here, so the
        # programme-aware category map is used directly.
        category = get_course_category(
            category_map=category_map,
            programme=programme,
            course_code=code,
        )

        if category in completed_counts:
            completed_counts[category] += 1

    remaining: Dict[str, Dict[str, Any]] = {}

    for rule in programme_rules:
        category = str(rule.rule_type).strip().upper()

        if category not in completed_counts:
            continue

        completed = completed_counts[category]

        # -----------------------------------------------------------
        # Unverified requirement
        # -----------------------------------------------------------

        if rule.min_required is None:
            remaining[category] = {
                "required": None,
                "completed": completed,
                "remaining": None,
                "status": "unverified",
                "description": rule.description,
            }

            continue

        # -----------------------------------------------------------
        # Verified numeric requirement
        # -----------------------------------------------------------

        required = int(rule.min_required)

        remaining[category] = {
            "required": required,
            "completed": completed,
            "remaining": max(required - completed, 0),
            "status": "verified",
            "description": rule.description,
        }

    # ---------------------------------------------------------------
    # If a category does not have a rule, do not invent one.
    # ---------------------------------------------------------------

    return remaining


# ---------------------------------------------------------------------------
# Prerequisite checking
# ---------------------------------------------------------------------------

def prerequisites_met(
    course: Course,
    completed_courses: List[str],
) -> bool:
    """
    Check whether all explicitly listed prerequisites are completed.

    If a course has no prerequisites, it passes automatically.

    This function intentionally only checks prerequisites explicitly
    represented in the processed Course object.
    """

    completed = {
        normalize_code(code)
        for code in completed_courses
    }

    prerequisites = getattr(course, "prerequisites", None)

    if not prerequisites:
        return True

    for prerequisite in prerequisites:

        # Handle string prerequisite.
        if isinstance(prerequisite, str):
            if normalize_code(prerequisite) not in completed:
                return False

        # Handle objects/dicts with a course_code/code field.
        elif isinstance(prerequisite, dict):
            prerequisite_code = (
                prerequisite.get("course_code")
                or prerequisite.get("code")
            )

            if prerequisite_code:
                if normalize_code(prerequisite_code) not in completed:
                    return False

        # Handle Pydantic-like prerequisite objects.
        else:
            prerequisite_code = getattr(
                prerequisite,
                "course_code",
                None,
            ) or getattr(
                prerequisite,
                "code",
                None,
            )

            if prerequisite_code:
                if normalize_code(prerequisite_code) not in completed:
                    return False

    return True


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def get_eligible_courses(
    profile: StudentProfile,
    courses: List[Course],
    rules: List[ProgrammeRule],
    category_map: Dict[str, Any],
) -> List[Course]:
    """
    Return courses that are deterministically eligible.

    A course is eligible only if:

    1. It has a known category.
    2. That category has a verified requirement.
    3. The student still has remaining slots in that category.
    4. The course is not already completed.
    5. Its prerequisites are satisfied.

    Unknown/unclassified courses are excluded.

    This is intentionally conservative. The recommender can later expose
    such courses as "requires verification" rather than silently calling
    them eligible.
    """

    remaining_requirements = compute_remaining_requirements(
        profile=profile,
        rules=rules,
        category_map=category_map,
    )

    completed = {
        normalize_code(code)
        for code in profile.completed_courses
    }

    eligible: List[Course] = []

    for course in courses:

        course_code = normalize_code(course.course_code)

        # -----------------------------------------------------------
        # Already completed
        # -----------------------------------------------------------

        if course_code in completed:
            continue

        # -----------------------------------------------------------
        # Determine category
        # -----------------------------------------------------------

        course_category = getattr(
            course,
            "category",
            None,
        )

        category = get_course_category(
            category_map=category_map,
            programme=profile.degree,
            course_code=course_code,
            course_category=course_category,
        )

        # Unknown category -> do not guess.
        if category is None:
            continue

        # -----------------------------------------------------------
        # Find requirement for category
        # -----------------------------------------------------------

        requirement = remaining_requirements.get(category)

        if requirement is None:
            continue

        # -----------------------------------------------------------
        # Unverified requirement -> not deterministically eligible
        # -----------------------------------------------------------

        if requirement.get("status") != "verified":
            continue

        # -----------------------------------------------------------
        # Requirement already satisfied
        # -----------------------------------------------------------

        if requirement.get("remaining", 0) <= 0:
            continue

        # -----------------------------------------------------------
        # Prerequisites
        # -----------------------------------------------------------

        if not prerequisites_met(
            course,
            profile.completed_courses,
        ):
            continue

        if course_code in {
            normalize_code(c.course_code)
            for c in eligible
        }:
            continue

        eligible.append(course)

    return eligible


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def print_requirements(
    remaining_requirements: Dict[str, Dict[str, Any]]
) -> None:
    """Pretty-print requirement information."""

    print("\n=== REMAINING REQUIREMENTS ===")

    print(json.dumps(
        remaining_requirements,
        indent=2,
        ensure_ascii=False,
    ))


def print_eligible_courses(
    eligible_courses: List[Course],
) -> None:
    """Print eligible courses."""

    print("\n=== ELIGIBLE COURSES ===")
    print(
        f"{len(eligible_courses)} eligible courses"
    )

    for course in eligible_courses:
        code = getattr(course, "course_code", "?")
        title = getattr(course, "title", "?")
        category = getattr(course, "category", None)

        print(
            f" - {code}: {title}"
            f" [{category or 'programme-mapped'}]"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main() -> None:

    print("Loading processed data...")

    courses = load_courses()
    rules = load_rules()
    category_map = load_category_map()

    print(f"Courses loaded: {len(courses)}")
    print(f"Programme rules loaded: {len(rules)}")
    print(f"Category map entries: {len(category_map)}")

    # ---------------------------------------------------------------
    # Test profile
    # ---------------------------------------------------------------

    profile = StudentProfile(
        campus="Pilani",
        admission_year=2025,
        degree="B.E. Computer Science",
        current_semester=3,
        completed_courses=["CS F211"],
        interests=[
            "machine learning",
            "artificial intelligence",
            "computer science",
        ],
        minor=None,
    )

    # ---------------------------------------------------------------
    # Category lookup tests
    # ---------------------------------------------------------------

    print("\n=== CATEGORY LOOKUPS ===")

    test_courses = [
        "CS F211",
        "CS F407",
        "HSS F222",
    ]

    for code in test_courses:
        category = get_course_category(
            category_map,
            profile.degree,
            code,
        )

        print(f"{code}: {category}")

    # ---------------------------------------------------------------
    # Requirement calculation
    # ---------------------------------------------------------------

    remaining_requirements = compute_remaining_requirements(
        profile=profile,
        rules=rules,
        category_map=category_map,
    )

    print_requirements(remaining_requirements)

    # ---------------------------------------------------------------
    # Eligibility
    # ---------------------------------------------------------------

    eligible = get_eligible_courses(
        profile=profile,
        courses=courses,
        rules=rules,
        category_map=category_map,
    )

    print_eligible_courses(eligible)


if __name__ == "__main__":
    main()