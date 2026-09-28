import streamlit as st

from schema import StudentProfile

from requirement_engine import (
    load_courses,
    load_rules,
    load_category_map,
    compute_remaining_requirements,
    get_eligible_courses,
    get_course_category,
)

from recommend import recommend


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="BITS Course Recommender",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def get_courses():
    return load_courses()


@st.cache_data
def get_rules():
    return load_rules()


@st.cache_data
def get_category_map():
    return load_category_map()


courses = get_courses()
rules = get_rules()
category_map = get_category_map()


PROGRAMME_NAMES = sorted(
    {
        r.programme
        for r in rules
        if getattr(r, "programme", None)
    }
)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 BITS Academic Course Recommender")

st.caption(
    f"{len(courses)} courses loaded · "
    f"{len(rules)} programme rules · "
    f"{len(category_map)} category mappings"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Recommendation Mode")

    use_gemini = st.toggle(
        "🤖 Use Gemini",
        value=False,
        help=(
            "Use Gemini to understand natural-language queries. "
            "If Gemini fails or is unavailable, the local parser is used automatically."
        ),
    )

    if use_gemini:
        st.caption("🤖 Gemini Enhanced")
    else:
        st.caption("⚡ Local — no API calls")

    st.divider()

    st.header("Student Profile")

    campus = st.text_input(
        "Campus",
        value="Pilani",
    )

    admission_year = st.number_input(
        "Admission year",
        min_value=2018,
        max_value=2026,
        value=2025,
    )

    degree = st.selectbox(
        "Degree / Programme",
        PROGRAMME_NAMES,
    )

    current_semester = st.number_input(
        "Current semester",
        min_value=1,
        max_value=8,
        value=3,
    )

    completed_raw = st.text_area(
        "Completed courses",
        value="CS F211",
        help="Comma-separated course codes",
    )

    current_raw = st.text_area(
        "Currently enrolled courses",
        value="",
        help="Comma-separated course codes",
    )

    minor = st.text_input(
        "Minor",
        value="",
    )

    interests_raw = st.text_input(
        "Interests",
        value="AI, machine learning, security",
    )

    st.divider()

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------






# ============================================================
# PROFILE
# ============================================================

profile = StudentProfile(
    campus=campus,
    admission_year=int(admission_year),
    degree=degree,
    current_semester=int(current_semester),

    completed_courses=[
        c.strip().upper()
        for c in completed_raw.split(",")
        if c.strip()
    ],

    current_courses=[
        c.strip().upper()
        for c in current_raw.split(",")
        if c.strip()
    ],

    minor=minor or None,

    interests=[
        i.strip()
        for i in interests_raw.split(",")
        if i.strip()
    ],
)


# ============================================================
# REQUIREMENTS
# ============================================================

remaining = compute_remaining_requirements(
    profile,
    rules,
    category_map,
)


st.subheader("Your Remaining Requirements")


if not remaining:

    st.warning(
        f"No programme rules found for '{degree}'."
    )

else:

    cols = st.columns(len(remaining))

    for col, (rule_type, info) in zip(
        cols,
        remaining.items(),
    ):

        with col:

            required = info.get("required")
            remaining_count = info.get("remaining")

            if (
                required is not None
                and remaining_count is not None
            ):

                st.metric(
                    rule_type,
                    f"{remaining_count} / {required} left",
                )

                st.caption(
                    "Verified from programme rules."
                )

            else:

                st.metric(
                    rule_type,
                    "Unverified",
                )

            description = info.get(
                "description",
                "",
            )

            st.caption(
                description[:160]
                + (
                    "..."
                    if len(description) > 160
                    else ""
                )
            )


# ============================================================
# QUERY
# ============================================================

st.divider()

st.subheader("Ask for a Recommendation")

st.caption(
    'Examples: "Suggest an AI-related DEL" · '
    '"AI DEL with no midsem" · '
    '"OPEL with no attendance requirement"'
)

query = st.text_input(
    "Your question",
    placeholder="Suggest an AI-related DEL with no midsem",
)


# ============================================================
# RESULTS
# ============================================================

if query:

    with st.spinner(
        "Checking requirements and finding matching courses..."
    ):

        try:

            results, parser_used = recommend(
                profile,
                query,
                courses,
                rules,
                category_map,
                top_k=5,
                use_gemini=use_gemini,
            )

        except Exception as e:

            results = None
            parser_used = None

            st.error(
                f"Recommendation failed: {e}"
            )

    if results is not None:

        st.caption(
            f"Intent parser used: **{parser_used}**"
        )

        if not results:

            st.info(
                "No eligible courses matched all "
                "the constraints in your query."
            )

        else:

            st.success(
                f"Found {len(results)} matching course"
                + (
                    "s"
                    if len(results) != 1
                    else ""
                )
                + "."
            )

            for i, result in enumerate(
                results,
                start=1,
            ):

                with st.container(border=True):

                    col1, col2, col3 = st.columns(
                        [0.55, 0.25, 0.20]
                    )

                    with col1:

                        st.markdown(
                            f"### {i}. "
                            f"{result['course_code']} — "
                            f"{result['title']}"
                        )

                    with col2:

                        st.metric(
                            "Category",
                            result.get(
                                "category",
                                "Unknown",
                            ),
                        )

                    with col3:

                        st.metric(
                            "Match",
                            f"{result.get('similarity', 0):.2f}",
                        )

                    st.write(
                        result["explanation"]
                    )


# ============================================================
# DEBUG
# ============================================================

st.divider()

with st.expander(
    "Debug: full eligible course list"
):

    eligible = get_eligible_courses(
        profile,
        courses,
        rules,
        category_map,
    )

    st.write(
        f"{len(eligible)} eligible courses"
    )

    st.dataframe(
        [
            {
                "code": c.course_code,
                "title": c.title,
                "category": get_course_category(
                    category_map,
                    profile.degree,
                    c.course_code,
                    getattr(c, "category", None),
                ),
            }
            for c in eligible
        ],
        use_container_width=True,
    )