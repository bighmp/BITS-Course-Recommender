"""
Extracts structured Course + HandoutData records from every handout PDF
in data/raw/handouts/ using regex + table parsing only. No LLM, no API
key, no rate limits, no cost — runs on all 540 files in well under a
minute.

v2 fixes over the first pass:
- Evaluation table: falls back to text-strategy table detection, then to
  raw-text line matching, when a PDF has no visible gridlines (common in
  Word->PDF exports) — this was why most files came back with an empty
  evaluation_components list.
- Topics: only reads the actual "Lecture Topics" column, instead of every
  column in the row (which was pulling in Learning Outcomes fragments
  like "Understanding of").
- Attendance: prefers an explicitly labeled "Attendance Policy:" section
  over the first sentence anywhere containing the word "attendance".
- Strips repeated page-footer boilerplate ("Please Do Not Print...",
  "Instructor-in-charge ...") that was leaking into makeup_policy etc.

Usage:
    python extract_handouts_free.py

Requires: pdfplumber, pydantic
    pip install pdfplumber pydantic
"""

import json
import re
from pathlib import Path

import pdfplumber

from schema import Course, HandoutData, SourceMeta

RAW_DIR = Path("data/raw/handouts")
OUT_PATH = Path("data/processed/courses.json")

DEPT_PREFIX_RE = re.compile(r"^([A-Z]+)")
COURSE_NO_RE = re.compile(
    r"Course\s*(?:No\.?|Number)\s*:?\s*\[?\s*(?:[A-Z]+\s*/\s*)*([A-Z]{2,6}\s?[A-Z]\d{2,4}[A-Z]?)",
    re.IGNORECASE,
)
COURSE_TITLE_RE = re.compile(r"Course\s*Title\s*:?\s*(.+)", re.IGNORECASE)
INSTRUCTOR_RE = re.compile(r"Instructor[- ]in[- ][Cc]harge\s*:?\s*(.+)", re.IGNORECASE)

ATTENDANCE_LABELED_RE = re.compile(
    r"Attendance\s*Policy\s*:?\s*(.+?)(?=\n\s*(?:\d+\.\s*[A-Z]|Notices|\Z))",
    re.IGNORECASE | re.DOTALL,
)
ATTENDANCE_FALLBACK_RE = re.compile(r"([^.\n]*attendance[^.\n]*\.)", re.IGNORECASE)

MAKEUP_SECTION_RE = re.compile(
    r"(?:\d+\.\s*)?Make-?up\s*Policy\s*:?\s*(.+?)(?=\n\s*\d+\.\s*[A-Z]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
DESCRIPTION_SECTION_RE = re.compile(
    r"Course\s*Description\s*:?\s*(.+?)(?=\n\s*\d+\.\s*[A-Z]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
EVAL_SECTION_RE = re.compile(
    r"(?:\d+\.\s*)?Evaluation\s*Scheme\s*:?\s*(.+?)(?=\n\s*\d+\.\s*[A-Z]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Page-footer boilerplate that bleeds into a section when that section
# runs to the bottom of a page — strip it wherever it shows up.
FOOTER_JUNK_RE = re.compile(
    r"\s*(Please\s*Do\s*Not\s*Print.*|Instructor-in-[Cc]harge\s*\(?[A-Z]{2,6}\s?[FG]?\d*\)?\s*_?|BIRLA\s+INSTITUTE.*)",
    re.IGNORECASE | re.DOTALL,
)


def clean(s: str) -> str:
    s = FOOTER_JUNK_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def tables_with_fallback(page) -> list:
    """Line-based detection (pdfplumber's default) misses tables with no
    visible gridlines. If it finds nothing, retry with text-based
    detection before giving up."""
    tables = page.extract_tables() or []
    if tables:
        return tables
    try:
        return page.extract_tables(table_settings={
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        }) or []
    except Exception:
        return []


def find_evaluation_rows(tables: list, full_text: str) -> tuple[list[str], bool]:
    """Try, in order: bordered/borderless table -> raw-text line match.
    Returns (rows, was_table) so the caller can set confidence accordingly.

    NOTE: the word "Evaluation" is often in a heading paragraph ABOVE the
    table (e.g. "6. Evaluation Scheme:"), not inside the table's own
    header row. Requiring "evaluation" in the header row itself missed
    real, well-formed tables headed "Component | Duration | Wtg. (%)..."
    So we match on the table's actual column vocabulary instead."""
    for table in tables:
        if not table or not table[0]:
            continue
        header = " ".join(c or "" for c in table[0]).lower()
        looks_like_eval_table = (
            ("component" in header or "evaluation" in header)
            and ("%" in header or "weightage" in header or "wtg" in header)
        )
        if looks_like_eval_table:
            rows = []
            for row in table[1:]:
                cells = [clean(c) for c in row if c]
                if cells:
                    rows.append(" - ".join(cells))
            if rows:
                return rows, True

    section = EVAL_SECTION_RE.search(full_text)
    if section:
        rows = [
            clean(line) for line in section.group(1).split("\n")
            if re.search(r"\d+\s*%", line) and len(line.strip()) > 5
        ]
        if rows:
            return rows[:10], False

    return [], False


def find_course_plan_topics(tables: list) -> list[str]:
    """Read ONLY the Lecture Topics column of the course-plan table —
    pulling every column mixes in Learning Outcomes text as if it were
    a topic, which is what the v1 version did wrong."""
    for table in tables:
        if not table or not table[0]:
            continue
        header_lower = [clean(c or "").lower() for c in table[0]]
        topics_col = next((i for i, h in enumerate(header_lower) if "topic" in h), None)
        if topics_col is None:
            continue

        topics = []
        for row in table[1:]:
            if topics_col >= len(row) or not row[topics_col]:
                continue
            # rejoin a wrapped cell into one string rather than treating
            # each wrapped line as its own separate "topic"
            merged = clean(row[topics_col].replace("\n", " "))
            if len(merged) > 15:
                topics.append(merged)
        if topics:
            return topics[:20]
    return []


def fix_repeated_char_artifact(text: str) -> str:
    triple_run_count = len(re.findall(r"([A-Za-z])\1{2,}", text))
    if triple_run_count < 15:
        return text  # not corrupted — leave legitimate triples like "EEE" alone
    return re.sub(r"([A-Za-z])\1{2,}", r"\1", text)


def extract_one(path: Path) -> dict:
    with pdfplumber.open(path) as pdf:
        full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        full_text = fix_repeated_char_artifact(full_text)
        all_tables = []
        for page in pdf.pages:
            all_tables.extend(tables_with_fallback(page))

    confidence = "high"
    notes = []

    course_code_m = COURSE_NO_RE.search(full_text)
    course_code = clean(course_code_m.group(1)).upper() if course_code_m else None
    if not course_code:
        confidence = "needs_verification"
        notes.append("course_code not found")

    title_m = COURSE_TITLE_RE.search(full_text)
    title = clean(title_m.group(1)) if title_m else None
    if not title:
        confidence = "needs_verification"
        notes.append("title not found")

    if not course_code or not title:
        return {"source_file": path.name, "error": "missing required fields: " + ", ".join(notes)}

    instructor_m = INSTRUCTOR_RE.search(full_text)
    instructor = clean(instructor_m.group(1)) if instructor_m else None

    dept_m = DEPT_PREFIX_RE.match(course_code)
    department = dept_m.group(1) if dept_m else "UNKNOWN"

    eval_rows, from_table = find_evaluation_rows(all_tables, full_text)
    if not eval_rows:
        notes.append("evaluation scheme not found (no table, no text match)")
        confidence = "medium" if confidence == "high" else confidence
    elif not from_table:
        notes.append("evaluation scheme found via text fallback, not a ruled table")
        confidence = "medium" if confidence == "high" else confidence

    eval_blob = " ".join(eval_rows).lower()
    has_midsem = (
        "mid-sem" in eval_blob or "mid sem" in eval_blob
        or "mid-term" in eval_blob or "midterm" in eval_blob
    ) if eval_rows else None
    has_compre = ("compre" in eval_blob) if eval_rows else None

    makeup_m = MAKEUP_SECTION_RE.search(full_text)
    makeup_policy = clean(makeup_m.group(1))[:600] if makeup_m else None

    attendance_m = ATTENDANCE_LABELED_RE.search(full_text) or ATTENDANCE_FALLBACK_RE.search(full_text)
    attendance_policy = clean(attendance_m.group(1)) if attendance_m else None

    topics = find_course_plan_topics(all_tables)
    if not topics:
        desc_m = DESCRIPTION_SECTION_RE.search(full_text)
        if desc_m:
            topics = [clean(desc_m.group(1))[:400]]
            notes.append("topics fell back to course description (no lecture-plan table found)")
            confidence = "medium" if confidence == "high" else confidence

    course = Course(
        course_code=course_code,
        title=title,
        department=department,
        category=None,  # not in handout Part-II; fill from bulletin/programme_rules later
        topics=topics,
        prerequisites=[],
        restrictions=[],
        handout=HandoutData(
            attendance_policy=attendance_policy,
            has_midsem=has_midsem,
            has_compre=has_compre,
            evaluation_components=eval_rows,
            makeup_policy=makeup_policy,
            instructor=instructor,
            topics=topics,
        ),
        source=SourceMeta(source_file=path.name, extraction_confidence=confidence),
    )
    return course.model_dump()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Processing {len(all_pdfs)} handouts (no API calls, this is fast)...")

    results = []
    for path in all_pdfs:
        try:
            results.append(extract_one(path))
        except Exception as e:
            results.append({"source_file": path.name, "error": str(e)})

    OUT_PATH.write_text(json.dumps(results, indent=2))

    succeeded = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    needs_review = [r for r in succeeded if r["source"]["extraction_confidence"] != "high"]
    no_eval = [r for r in succeeded if not r["handout"]["evaluation_components"]]

    print(f"\nDone: {len(succeeded)} extracted, {len(failed)} failed entirely.")
    print(f"{len(needs_review)} are medium/needs_verification confidence.")
    print(f"{len(no_eval)} have no evaluation_components at all.")
    if failed:
        print("Failed:", [f["source_file"] for f in failed[:20]], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()

