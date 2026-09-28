"""
Extract programme-specific CDC/DEL and global HUEL course categories
from the BITS bulletin PDF.

Usage:
    python scripts/extract_category_map.py data/raw/bulletin-314-336.pdf

Output:
    data/processed/course_category_map.json

Output structure:
{
    "<course_code>": "CDC",             # legacy flat CDC map
    ...
    "__GLOBAL_CATEGORIES__": {
        "HUEL": [...]
    },
    "__PROGRAMME_CATEGORIES__": {
        "B.E. Computer Science": {
            "CDC": [...],
            "DEL": [...]
        },
        ...
    }
}

IMPORTANT:
- Programme names are matched using explicit aliases.
- No fuzzy matching is used for programme names.
- The PDF is two-column, so each page is parsed left-column then
  right-column while preserving parser state across columns/pages.
- Unknown/unmatched programme headings are NOT silently merged.
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber


# ============================================================
# PATHS
# ============================================================

RULES_PATH = Path("data/processed/programme_rules.json")
OUTPUT_PATH = Path("data/processed/course_category_map.json")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for robust exact comparisons.

    This is intentionally conservative:
    - Unicode dashes -> '-'
    - repeated whitespace -> single space
    - case-insensitive
    """
    if not text:
        return ""

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("-", "-")
    text = text.replace("−", "-")

    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def normalize_code(code: str) -> str:
    return " ".join(code.upper().split())


# ============================================================
# SOURCE PROGRAMME -> OUTPUT PROGRAMME
# ============================================================
#
# These are explicit mappings from the headings actually appearing
# in bulletin-314-336.pdf to the names used by the recommender.
#
# NO FUZZY MATCHING HERE.
# ============================================================

PROGRAMME_ALIASES = {
    # --------------------------------------------------------
    # B.E.
    # --------------------------------------------------------

    "ARCHITECTURAL AND URBAN ENGINEERING":
        "B.E. Architecture and Urban Engineering",

    "BIOTECHNOLOGY":
        "B.E. Biotechnology",

    "BIOTECHNOLOGY WITH SPECIALIZATION IN APPLIED MOLECULAR BIOLOGY":
        "B.E. Biotechnology with Specialization in Applied Molecular Biology",

    "CHEMICAL ENGINEERING":
        "B.E. Chemical",

    "CHEMICAL ENGINEERING WITH SPECIALIZATION IN ENERGY, ENVIRONMENT, AND SUSTAINABILITY":
        "B.E. Chemical with Specialization in Energy, Environment, and Sustainability",

    "CIVIL ENGINEERING":
        "B.E. Civil",

    "COMPUTER SCIENCE":
        "B.E. Computer Science",

    "ELECTRICAL AND ELECTRONICS ENGINEERING":
        "B.E. Electrical & Electronics",

    "ELECTRONICS AND COMMUNICATION ENGINEERING":
        "B.E. Electronics & Communication",

    "ELECTRONICS AND COMPUTER ENGINEERING":
        "B.E. Electronics & Computer Engineering",

    "ELECTRONICS AND INSTRUMENTATION ENGINEERING":
        "B.E. Electronics and Instrumentation",

    "ENVIRONMENTAL AND SUSTAINABILITY ENGINEERING":
        "B.E. Environmental and Sustainability Engineering",

    "MANUFACTURING ENGINEERING":
        "B.E. Manufacturing",

    "MATHEMATICS AND COMPUTING":
        "B.E. Mathematics and Computing",

    "MECHANICAL ENGINEERING":
        "B.E. Mechanical",

    "MECHANICAL ENGINEERING WITH SPECIALIZATION IN AEROSPACE":
        "B.E. Mechanical Engineering with Specialization in Aerospace",

    "PHARMACY":
        "B.Pharm.",

    "ROBOTICS AND INDUSTRIAL AUTOMATION":
        "B.E. Robotics and Industrial Automation",

    # --------------------------------------------------------
    # GENERAL STUDIES
    # --------------------------------------------------------

    "GENERAL STUDIES - COMMUNICATION AND MEDIA STUDIES STREAM":
        "GENERAL STUDIES - COMMUNICATION AND MEDIA STUDIES STREAM",

    "GENERAL STUDIES - DEVELOPMENT STUDIES STREAM":
        "GENERAL STUDIES - DEVELOPMENT STUDIES STREAM",

    # --------------------------------------------------------
    # M.Sc.
    # --------------------------------------------------------

    "BIOLOGICAL SCIENCES":
        "M.Sc. Biological Sciences",

    "CHEMISTRY":
        "M.Sc. Chemistry",

    "ECONOMICS":
        "M.Sc. Economics",

    "MATHEMATICS":
        "M.Sc. Mathematics",

    "PHYSICS":
        "M.Sc. Physics",

    "PHYSICS WITH SPECIALIZATION IN SPACE SCIENCE AND TECHNOLOGY":
        "M.Sc. Physics with Specialization in Space Science and Technology",

    "SEMICONDUCTOR AND NANOSCIENCE":
        "M.Sc. Semiconductor and Nanoscience",

    # --------------------------------------------------------
    # BBA
    # --------------------------------------------------------

    "BACHELOR OF BUSINESS ADMINISTRATION (HONOURS)":
        "BBA (Honours)",
}


# Normalize aliases once so matching is consistent.
NORMALIZED_PROGRAMME_ALIASES = {
    normalize_text(source): target
    for source, target in PROGRAMME_ALIASES.items()
}


def canonicalize_programme(raw_heading: str) -> str:
    """
    Convert a source PDF programme heading into our canonical name.

    There is deliberately NO fuzzy matching.
    """
    key = normalize_text(raw_heading)

    if key in NORMALIZED_PROGRAMME_ALIASES:
        return NORMALIZED_PROGRAMME_ALIASES[key]

    # Unknown heading: preserve it exactly rather than guessing.
    return re.sub(r"\s+", " ", raw_heading).strip()


# ============================================================
# PROGRAMME RULES
# ============================================================

def load_known_programmes():
    """
    Load programme names from programme_rules.json.

    This is used only for reporting / diagnostics.
    The bulletin extraction itself is controlled by explicit aliases.
    """
    if not RULES_PATH.exists():
        return []

    try:
        raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not read {RULES_PATH}: {exc}")
        return []

    names = set()

    for item in raw:
        if not isinstance(item, dict):
            continue

        programme = item.get("programme")

        if isinstance(programme, str) and programme.strip():
            names.add(programme.strip())

    return sorted(names)


# ============================================================
# COURSE CODE EXTRACTION
# ============================================================

# Examples:
#   CS F407
#   BITS F471
#   EEE G513
#   INSTR F424
#   SNS F312
#
# Department codes are generally 2-6 alphanumeric characters.
COURSE_CODE_RE = re.compile(
    r"\b[A-Z][A-Z0-9]{1,5}\s+[FGH]\d{3}\b"
)


def extract_course_codes(line: str):
    """
    Extract all course codes appearing on a line.

    Multiple codes are possible because the bulletin contains
    alternatives such as:
        MATH F212 OR ME F344
    """
    matches = COURSE_CODE_RE.findall(line)

    return [normalize_code(m) for m in matches]


# ============================================================
# PROGRAMME HEADING DETECTION
# ============================================================

# ============================================================
# SECTION / PROGRAMME HEADING DETECTION
# ============================================================

def is_core_heading(line: str) -> bool:
    """
    Detect:
        CORE COURSES
        CORE COURSES L P U
    """
    text = normalize_text(line)
    return text.startswith("CORE COURSES")


def is_del_heading(line: str) -> bool:
    """
    Detect:
        DISCIPLINE ELECTIVE COURSES
        DISCIPLINE ELECTIVE COURSES L P U
    """
    text = normalize_text(line)
    return text.startswith("DISCIPLINE ELECTIVE COURSES")


def is_huel_heading(line: str) -> bool:
    """
    Detect the start of the Humanities Elective pool.
    """
    text = normalize_text(line)

    return (
        "POOL OF HUMANITIES COURSES FOR FIRST DEGREE PROGRAMMES"
        in text
    )


def huel_heading_span(lines, index):
    """
    Detect the HUEL heading even when pdfplumber splits it
    across multiple lines.
    """

    target = "POOL OF HUMANITIES COURSES FOR FIRST DEGREE PROGRAMMES"

    for width in range(1, 5):

        if index + width > len(lines):
            break

        candidate = " ".join(
            line.strip()
            for line in lines[index:index + width]
            if line.strip()
        )

        if target in normalize_text(candidate):
            return width

    return 0


def is_huel_stop_heading(line: str) -> bool:
    text = normalize_text(line)

    return (
        text.startswith("OTHER COURSES")
        or text.startswith("LIST OF AUDIT TYPE COURSES")
    )


def is_project_type_heading(line: str) -> bool:
    return "PROJECT TYPE COURSES" in normalize_text(line)


def looks_like_programme_heading(line: str) -> bool:
    """
    Real programme headings in the bulletin are uppercase.

    Prevents course-title continuation lines such as:

        Computer Science
        Chemical Engineering
        Nanotechnology

    from being mistaken for programme headings.
    """

    text = line.strip()

    if not text:
        return False

    # Don't treat course-code lines as programme headings.
    if COURSE_CODE_RE.search(text):
        return False

    letters = "".join(
        ch for ch in text
        if ch.isalpha()
    )

    if not letters:
        return False

    return letters == letters.upper()


def find_programme_heading(lines, index):
    """
    Detect a real programme heading.

    Programme headings are uppercase in the bulletin, so only
    uppercase candidate lines are considered.
    """

    if not looks_like_programme_heading(lines[index]):
        return None, 0

    for width in range(1, 5):

        if index + width > len(lines):
            break

        candidate_lines = lines[index:index + width]

        # Every line of a multi-line heading must itself be uppercase.
        if not all(
            looks_like_programme_heading(line)
            for line in candidate_lines
        ):
            break

        candidate = " ".join(
            line.strip()
            for line in candidate_lines
            if line.strip()
        )

        normalized = normalize_text(candidate)

        if normalized in NORMALIZED_PROGRAMME_ALIASES:
            return (
                NORMALIZED_PROGRAMME_ALIASES[normalized],
                width,
            )

    return None, 0





# ============================================================
# COLLISION CHECK
# ============================================================

def check_programme_collision(
    source_for_canonical,
    canonical,
    raw_heading,
):
    """
    Prevent two different source headings from silently mapping
    to the same canonical programme.
    """

    raw_clean = re.sub(r"\s+", " ", raw_heading).strip()

    previous = source_for_canonical.get(canonical)

    if previous is None:
        source_for_canonical[canonical] = raw_clean
        return

    if normalize_text(previous) != normalize_text(raw_clean):
        raise ValueError(
            "\nProgramme collision detected!\n"
            f"  Canonical: {canonical}\n"
            f"  First source heading:  {previous}\n"
            f"  Second source heading: {raw_clean}\n"
            "\n"
            "This usually means an alias is incorrect.\n"
            "Fix PROGRAMME_ALIASES before continuing."
        )


# ============================================================
# MAIN PDF EXTRACTION
# ============================================================

def extract_appendix(pdf_path: Path):
    """
    Parse the two-column bulletin.

    The bulletin is laid out in two columns, so each page is split
    into left and right columns and parsed independently.

    Programme headings are used to determine the current programme.
    CORE COURSES activates CDC extraction.
    DISCIPLINE ELECTIVE COURSES activates DEL extraction.
    The Humanities pool is extracted globally.

    Important:
    If a CORE COURSES heading is encountered without a programme
    heading immediately preceding it, we do NOT inherit the previous
    programme. This prevents the BBA section from being incorrectly
    added to Semiconductor & Nanoscience.
    """

    programme_categories = {}

    global_categories = {
        "HUEL": set(),
    }

    source_for_canonical = {}

    current_programme = None
    current_category = None

    huel_active = False
    project_section_active = False

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            midpoint = page.width / 2

            columns = [
                (
                    "left",
                    page.crop(
                        (0, 0, midpoint, page.height)
                    ),
                ),
                (
                    "right",
                    page.crop(
                        (midpoint, 0, page.width, page.height)
                    ),
                ),
            ]

            for column_name, column in columns:

                text = column.extract_text(
                    x_tolerance=2,
                    y_tolerance=3,
                ) or ""

                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                i = 0

                while i < len(lines):

                    line = lines[i]

                    # ==================================================
                    # HUEL STOP
                    # ==================================================

                    if (
                        huel_active
                        and is_huel_stop_heading(line)
                    ):

                        huel_active = False
                        current_programme = None
                        current_category = None
                        project_section_active = False

                        i += 1
                        continue

                    # ==================================================
                    # HUEL START
                    # ==================================================

                    huel_span = huel_heading_span(
                        lines,
                        i,
                    )

                    if huel_span:

                        huel_active = True
                        current_programme = None
                        current_category = None
                        project_section_active = False

                        i += huel_span
                        continue

                    # ==================================================
                    # PROGRAMME HEADING
                    # ==================================================

                    programme, consumed = find_programme_heading(
                        lines,
                        i,
                    )

                    if programme is not None:

                        check_programme_collision(
                            source_for_canonical,
                            programme,
                            " ".join(
                                lines[
                                    i:i + consumed
                                ]
                            ),
                        )

                        current_programme = programme

                        programme_categories.setdefault(
                            current_programme,
                            {
                                "CDC": set(),
                                "DEL": set(),
                            },
                        )

                        current_category = None
                        huel_active = False
                        project_section_active = False

                        i += consumed
                        continue

                    # ==================================================
                    # PROJECT TYPE COURSES
                    # ==================================================

                    if is_project_type_heading(line):

                        project_section_active = True
                        current_programme = None
                        current_category = None
                        huel_active = False

                        i += 1
                        continue

                    # ==================================================
                    # CORE COURSES
                    # ==================================================

                    if is_core_heading(line):

                        current_category = None
                        project_section_active = False

                        # ------------------------------------------------
                        # Find a programme heading immediately before this
                        # CORE COURSES heading.
                        #
                        # We search backwards a few lines because some
                        # programme headings span multiple lines.
                        # ------------------------------------------------

                        recent_programme = None

                        for back in range(1, 5):

                            j = i - back

                            if j < 0:
                                break

                            candidate, _ = find_programme_heading(
                                lines,
                                j,
                            )

                            if candidate is not None:

                                recent_programme = candidate
                                break

                        # ------------------------------------------------
                        # Only activate CDC if a programme heading was
                        # actually found nearby.
                        # ------------------------------------------------

                        if recent_programme is not None:

                            current_programme = recent_programme

                            programme_categories.setdefault(
                                current_programme,
                                {
                                    "CDC": set(),
                                    "DEL": set(),
                                },
                            )

                            current_category = "CDC"

                        else:

                            # Do NOT carry the previous programme forward.
                            current_programme = None
                            current_category = None

                        i += 1
                        continue

                    # ==================================================
                    # DISCIPLINE ELECTIVES
                    # ==================================================

                    if is_del_heading(line):

                        if current_programme is not None:

                            current_category = "DEL"
                            project_section_active = False

                        i += 1
                        continue

                    # ==================================================
                    # COURSE CODES
                    # ==================================================

                    codes = extract_course_codes(line)

                    if codes:

                        # ------------------------------------------------
                        # GLOBAL HUEL
                        # ------------------------------------------------

                        if huel_active:

                            for code in codes:
                                global_categories["HUEL"].add(code)

                        # ------------------------------------------------
                        # PROGRAMME CDC / DEL
                        # ------------------------------------------------

                        elif (
                            current_programme is not None
                            and current_category in {
                                "CDC",
                                "DEL",
                            }
                            and not project_section_active
                        ):

                            for code in codes:

                                programme_categories[
                                    current_programme
                                ][current_category].add(code)

                    i += 1

    return (
        programme_categories,
        global_categories,
    )

# ============================================================
# OUTPUT
# ============================================================

def build_output(programme_categories, global_categories):
    """
    Build JSON output.

    The nested programme-aware structure is the authoritative
    structure.

    The flat CDC map is retained only for backwards compatibility
    with older code.
    """

    output = {}

    # ------------------------------------------------------------
    # Legacy flat CDC map
    # ------------------------------------------------------------
    #
    # NOTE:
    # This is NOT programme-aware.
    # The recommender should use __PROGRAMME_CATEGORIES__ instead.
    #

    for programme, categories in programme_categories.items():

        for code in categories["CDC"]:

            # Do not overwrite an existing mapping.
            # CDC is the only legacy flat category retained.
            output.setdefault(code, "CDC")

    # ------------------------------------------------------------
    # Global categories
    # ------------------------------------------------------------

    output["__GLOBAL_CATEGORIES__"] = {
        category: sorted(courses)
        for category, courses in global_categories.items()
    }

    # ------------------------------------------------------------
    # Programme-specific categories
    # ------------------------------------------------------------

    output["__PROGRAMME_CATEGORIES__"] = {}

    for programme in sorted(programme_categories):

        output["__PROGRAMME_CATEGORIES__"][programme] = {
            "CDC": sorted(
                programme_categories[programme]["CDC"]
            ),
            "DEL": sorted(
                programme_categories[programme]["DEL"]
            ),
        }

    return output


def write_output(output):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# LOOKUP HELPERS FOR SMOKE TESTS
# ============================================================

def get_programme_categories(output, programme):
    programmes = output.get(
        "__PROGRAMME_CATEGORIES__",
        {},
    )

    # Exact normalized comparison so:
    # "B. E. Computer Science"
    # and
    # "B.E. Computer Science"
    # can be handled if necessary.
    target = normalize_text(programme)

    for name, categories in programmes.items():

        if normalize_text(name) == target:
            return categories

    return None


def code_in_category(output, programme, category, code):
    categories = get_programme_categories(
        output,
        programme,
    )

    if categories is None:
        return False

    return normalize_code(code) in {
        normalize_code(c)
        for c in categories.get(category, [])
    }


# ============================================================
# SMOKE TESTS
# ============================================================

def run_smoke_tests(output):

    print("\n=== SMOKE TESTS ===")

    tests = [
        "B.E. Computer Science",
        "B.E. Civil",
        "B.E. Electrical & Electronics",
        "B.E. Electronics & Communication",
        "B.E. Mathematics and Computing",
        "B.E. Environmental and Sustainability Engineering",
        "M.Sc. Semiconductor and Nanoscience",
    ]

    for programme in tests:

        categories = get_programme_categories(
            output,
            programme,
        )

        if categories is None:
            print(f"{programme}: NOT FOUND")
        else:
            print(
                f"{programme}: "
                f"CDC={len(categories.get('CDC', []))}, "
                f"DEL={len(categories.get('DEL', []))}"
            )

    # --------------------------------------------------------
    # CS
    # --------------------------------------------------------

    print(
        "CS F407 in CS DEL:",
        code_in_category(
            output,
            "B.E. Computer Science",
            "DEL",
            "CS F407",
        ),
    )

    print(
        "CS F407 in CS CDC:",
        code_in_category(
            output,
            "B.E. Computer Science",
            "CDC",
            "CS F407",
        ),
    )

    # --------------------------------------------------------
    # Civil
    # --------------------------------------------------------

    print(
        "CE F417 in Civil DEL:",
        code_in_category(
            output,
            "B.E. Civil",
            "DEL",
            "CE F417",
        ),
    )

    # --------------------------------------------------------
    # Global HUEL
    # --------------------------------------------------------

    huel = output.get(
        "__GLOBAL_CATEGORIES__",
        {},
    ).get("HUEL", [])

    print(
        "HSS F222 in global HUEL:",
        normalize_code("HSS F222") in {
            normalize_code(c)
            for c in huel
        },
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_output(output):
    """
    Basic structural checks.

    These don't enforce programme-specific academic rules;
    they simply catch obvious extraction corruption.
    """

    programmes = output.get(
        "__PROGRAMME_CATEGORIES__",
        {},
    )

    if not programmes:
        raise ValueError(
            "No programmes were detected. "
            "Check the PDF and parser."
        )

    huel = output.get(
        "__GLOBAL_CATEGORIES__",
        {},
    ).get("HUEL", [])

    if not huel:
        raise ValueError(
            "No HUEL courses were detected."
        )

    # CS should definitely exist.
    cs = get_programme_categories(
        output,
        "B.E. Computer Science",
    )

    if cs is None:
        raise ValueError(
            "B.E. Computer Science was not detected."
        )

    # CS F407 is explicitly a CS DEL in the bulletin.
    if "CS F407" not in cs["DEL"]:
        raise ValueError(
            "CS F407 is missing from B.E. Computer Science DEL."
        )

    # It must NOT simultaneously be CS CDC.
    if "CS F407" in cs["CDC"]:
        raise ValueError(
            "CS F407 was incorrectly classified as CS CDC."
        )

    # Civil sanity check.
    civil = get_programme_categories(
        output,
        "B.E. Civil",
    )

    if civil is None:
        raise ValueError(
            "B.E. Civil was not detected."
        )

    if "CE F417" not in civil["DEL"]:
        raise ValueError(
            "CE F417 is missing from B.E. Civil DEL."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python scripts/extract_category_map.py "
            "data/raw/bulletin-314-336.pdf"
        )
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    known_programmes = load_known_programmes()

    print(
        f"Known programme names: "
        f"{len(known_programmes)}"
    )

    try:

        programme_categories, global_categories = (
            extract_appendix(pdf_path)
        )

        output = build_output(
            programme_categories,
            global_categories,
        )

        validate_output(output)

        write_output(output)

    except Exception as exc:

        print("\nERROR while extracting category map:")
        print(exc)

        sys.exit(1)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(
        "Programmes detected:",
        len(
            output["__PROGRAMME_CATEGORIES__"]
        ),
    )

    print(
        "HUEL courses:",
        len(
            output[
                "__GLOBAL_CATEGORIES__"
            ]["HUEL"]
        ),
    )

    print()

    for programme, categories in sorted(
        output["__PROGRAMME_CATEGORIES__"].items()
    ):
        print(
            f"{programme}: "
            f"CDC={len(categories['CDC'])}, "
            f"DEL={len(categories['DEL'])}"
        )

    run_smoke_tests(output)


if __name__ == "__main__":
    main()