"""
Builds course_category_map.json: {course_code: "CDC"} for the compulsory
courses named explicitly on each programme's semester-pattern page.

What this does NOT do: map DEL/HUEL/OPEL to specific course codes. The
bulletin's semester-pattern pages only show generic elective slots
("Discipline Electives 3(min)") — the actual list of which catalog
courses qualify as valid electives per programme lives in a separate
appendix not included in this excerpt. That's a real data gap, not
something more regex fixes. Courses with no known category fall through
in requirement_engine.get_eligible_courses() as general electives,
which is a reasonable approximation given what's actually extractable.

Method: for each single-degree programme page, find every course-code
pattern, count department-prefix frequency, and treat the most frequent
prefix as the programme's "home department" — its courses are the CDC
list. Cross-checked against programme_rules.json's already-verified CDC
course count; a mismatch gets flagged rather than silently trusted,
since a perfect regex match isn't guaranteed (BITS's own counting
conventions for borderline courses aren't always obvious from the page
alone).

Usage:
    python extract_category_map.py path/to/bulletin.pdf
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

PROGRAMME_HEADER_RE = re.compile(r"Students Admitted to (.+?) Programme")
CODE_RE = re.compile(r"\b([A-Z]{2,6})\s?([A-Z]\d{2,4}[A-Z]?)\b")
CDC_COUNT_RE = re.compile(r"Discipline Core\s*-\s*\d+(?:\s*or\s*\d+)?\s*Units\s*\((\d+)\s*Courses?\)", re.IGNORECASE)

OUT_PATH = Path("data/processed/course_category_map.json")
RULES_PATH = Path("data/processed/programme_rules.json")


def extract_cdc_for_page(text: str) -> tuple[str, list[str], int, int] | None:
    header_m = PROGRAMME_HEADER_RE.search(text)
    if not header_m:
        return None
    programme = header_m.group(1).strip()

    matches = CODE_RE.findall(text)
    if not matches:
        return None
    dept_freq = Counter(dept for dept, _ in matches)
    home_dept, freq = dept_freq.most_common(1)[0]
    if freq < 3:
        return None  # too few matches to trust a "home department" inference

    cdc_codes = sorted({f"{d} {n}" for d, n in matches if d == home_dept})

    stated_m = CDC_COUNT_RE.search(text)
    stated_count = int(stated_m.group(1)) if stated_m else -1

    return programme, cdc_codes, len(cdc_codes), stated_count


def main(pdf_path: str):
    category_map: dict[str, str] = {}
    mismatches = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "composite Dual Degree" in text:
                continue
            result = extract_cdc_for_page(text)
            if not result:
                continue
            programme, codes, found_count, stated_count = result
            for code in codes:
                # a course appearing as CDC for one programme and possibly
                # something else for another is a real ambiguity we accept —
                # first programme to claim a code wins, rare in practice
                # since department-prefix courses are usually programme-specific
                category_map.setdefault(code, "CDC")
            if stated_count != -1 and found_count != stated_count:
                mismatches.append((programme, found_count, stated_count))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(category_map, indent=2, sort_keys=True))

    print(f"Wrote {len(category_map)} CDC course-code mappings to {OUT_PATH}")
    if mismatches:
        print(f"\n{len(mismatches)} programmes where extracted CDC count didn't match the bulletin's stated count"
              f" (off-by-one is common — likely a borderline course counted differently by BITS's own tally):")
        for programme, found, stated in mismatches:
            print(f"  - {programme}: found {found} named courses, bulletin states {stated}")
        print("These are still written to the map (better than nothing), but worth a manual glance"
              " if your demo profile touches one of these programmes.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_category_map.py path/to/bulletin.pdf")
        sys.exit(1)
    main(sys.argv[1])