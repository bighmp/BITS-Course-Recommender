"""
The deterministic core of the recommender. Given a student profile and
the programme's rules, this computes what's still required and filters
the full course catalog down to what the student is actually eligible
to take — BEFORE any preference/interest matching happens.

This is intentionally plain Python with no LLM involved anywhere. The
spec is explicit that eligibility must be deterministic and traceable;
this module is what makes that true.

Two inputs you need to populate by hand (see the bottom of this file
for a starter template):
  - data/processed/programme_rules.json  — CDC/DEL/HUEL/OPEL minimums
    per programme (scope this to 1-2 programmes, not every one BITS
    offers — see the note at the bottom).
  - data/processed/course_category_map.json — {course_code: category}
    for the courses relevant to whichever programme(s) you're testing.
    Your handout extraction doesn't produce `category` (it's not stated
    in Part-II handouts, only in the bulletin), so this fills that gap
    without requiring you to hand-parse the entire 33MB bulletin.

Usage:
    from requirement_engine import compute_remaining_requirements, get_eligible_courses
"""

import json
import re
from pathlib import Path

from schema import Course, ProgrammeRule, StudentProfile

COURSES_PATH = Path("data/processed/courses.json")
RULES_PATH = Path("data/processed/programme_rules.json")
CATEGORY_MAP_PATH = Path("data/processed/course_category_map.json")


def normalize_programme(name: str) -> str:
    """The bulletin itself is inconsistent — 'B. E. Computer Science' vs
    'B.E. Electrical & Electronics' — so compare on normalized form
    rather than exact string match, or a real degree name mismatch
    would silently return zero rules instead of erroring loudly."""
    return re.sub(r"\s+", " ", name.replace(".", ". ")).strip().lower()


def load_courses() -> list[Course]:
    raw = json.loads(COURSES_PATH.read_text())
    return [Course.model_validate(c) for c in raw if "error" not in c]


def load_rules() -> list[ProgrammeRule]:
    if not RULES_PATH.exists():
        return []
    raw = json.loads(RULES_PATH.read_text())
    return [ProgrammeRule.model_validate(r) for r in raw]


def load_category_map() -> dict[str, str]:
    if not CATEGORY_MAP_PATH.exists():
        return {}
    return json.loads(CATEGORY_MAP_PATH.read_text())


def normalize_code(code: str) -> str:
    """'BIO F311', 'BIO  F311', 'bio f311' all mean the same course —
    normalize so completed_courses/prerequisites comparisons don't
    silently fail on whitespace/case differences."""
    return " ".join(code.upper().split())


def compute_remaining_requirements(
    profile: StudentProfile,
    rules: list[ProgrammeRule],
    category_map: dict[str, str],
) -> dict[str, dict]:
    """For each rule_type (CDC/DEL/HUEL/OPEL) applicable to the
    student's programme, count how many the student has already
    completed (via category_map) and how many are still needed.

    Returns: {rule_type: {"required": int, "completed": int, "remaining": int, "description": str}}
    """
    completed = {normalize_code(c) for c in profile.completed_courses}

    # tally completed courses by category
    completed_by_category: dict[str, int] = {}
    for code in completed:
        cat = category_map.get(code)
        if cat:
            completed_by_category[cat] = completed_by_category.get(cat, 0) + 1

    result = {}
    for rule in rules:
        if normalize_programme(rule.programme) != normalize_programme(profile.degree):
            continue
        if rule.batch_scope and str(profile.admission_year) not in rule.batch_scope:
            continue  # rule doesn't apply to this student's batch

        required = rule.min_required or 0
        done = completed_by_category.get(rule.rule_type, 0)
        result[rule.rule_type] = {
            "required": required,
            "completed": done,
            "remaining": max(required - done, 0),
            "description": rule.description,
        }
    return result


def prerequisites_met(course: Course, completed: set[str]) -> bool:
    if not course.prerequisites:
        return True
    return all(normalize_code(p) in completed for p in course.prerequisites)


def get_eligible_courses(
    profile: StudentProfile,
    courses: list[Course],
    remaining_requirements: dict[str, dict],
    category_map: dict[str, str],
) -> list[Course]:
    """The core filter: a course is eligible only if ALL of:
      1. not already completed
      2. not currently enrolled in
      3. prerequisites satisfied
      4. EITHER it has no category (open elective-ish), OR its category
         still has remaining slots per remaining_requirements

    This is deliberately conservative — it's meant to narrow the field
    for the LLM/embedding layer to explain choices from, not to be the
    final word on every institute rule (restrictions like "not open to
    X branch" still need per-course handling if your test profiles hit
    them).
    """
    completed = {normalize_code(c) for c in profile.completed_courses}
    current = {normalize_code(c) for c in profile.current_courses}
    open_categories = {cat for cat, info in remaining_requirements.items() if info["remaining"] > 0}

    eligible = []
    for course in courses:
        code = normalize_code(course.course_code)
        if code in completed or code in current:
            continue
        if not prerequisites_met(course, completed):
            continue

        category = category_map.get(code) or course.category
        # no known category -> treat as an open/general elective, don't block it
        if category and category not in open_categories:
            continue

        eligible.append(course)
    return eligible


if __name__ == "__main__":
    # Quick smoke test with a made-up profile — replace with a real one
    # once programme_rules.json and course_category_map.json exist.
    courses = load_courses()
    rules = load_rules()
    category_map = load_category_map()

    test_profile = StudentProfile(
        campus="Pilani",
        admission_year=2023,
        degree="B.E. Computer Science",
        current_semester=5,
        completed_courses=["CS F211", "MATH F211"],
        current_courses=[],
        interests=["AI", "security"],
    )

    remaining = compute_remaining_requirements(test_profile, rules, category_map)
    print("Remaining requirements:", json.dumps(remaining, indent=2))

    eligible = get_eligible_courses(test_profile, courses, remaining, category_map)
    print(f"\n{len(eligible)} eligible courses out of {len(courses)} total.")
    for c in eligible[:10]:
        print(" -", c.course_code, c.title)

# --- programme_rules.json starter template ---
# Create data/processed/programme_rules.json with entries like this,
# scoped to 1-2 programmes (per earlier advice, don't try to cover
# every BITS programme):
#
# [
#   {
#     "programme": "B.E. Computer Science",
#     "rule_type": "DEL",
#     "description": "Department Elective — 4 courses required",
#     "min_required": 4,
#     "batch_scope": null,
#     "source": {"source_file": "bulletin.pdf (manual)", "extraction_confidence": "high"}
#   },
#   {
#     "programme": "B.E. Computer Science",
#     "rule_type": "HUEL",
#     "description": "Humanities elective — 2 courses required",
#     "min_required": 2,
#     "batch_scope": null,
#     "source": {"source_file": "bulletin.pdf (manual)", "extraction_confidence": "high"}
#   }
# ]
#
# --- course_category_map.json starter template ---
# {"CS F211": "CDC", "CS F215": "DEL", "HSS F223": "HUEL", ...}
# Only needs entries for courses your test profile could plausibly hit
# — not all 540. Grow it as you test more queries.