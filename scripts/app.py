"""
The dashboard. Profile form on the left, query box + results on the
right. Wires together everything built so far:
    schema.py           -> StudentProfile shape
    requirement_engine   -> remaining CDC/DEL/HUEL/OPEL, eligible courses
    recommend.py         -> natural-language query -> ranked, explained results

Run:
    streamlit run scripts/app.py
"""

import streamlit as st

from schema import StudentProfile
from requirement_engine import load_courses, load_rules, load_category_map, compute_remaining_requirements
from recommend import recommend

st.set_page_config(page_title="BITS Course Recommender", layout="wide")


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

PROGRAMME_NAMES = sorted({r.programme for r in rules}) or ["B.E. Computer Science"]

st.title("BITS Academic Course Recommender")
st.caption(f"{len(courses)} courses loaded · {len(rules)} programme rules · "
           f"{len(category_map)} courses category-mapped")

# --- Sidebar: student profile ---
with st.sidebar:
    st.header("Your Profile")

    campus = st.text_input("Campus", value="Pilani")
    admission_year = st.number_input("Admission year", min_value=2018, max_value=2026, value=2023)
    degree = st.selectbox("Degree / Programme", PROGRAMME_NAMES)
    current_semester = st.number_input("Current semester", min_value=1, max_value=8, value=5)

    completed_raw = st.text_area(
        "Completed courses (comma-separated course codes)",
        value="CS F211, CS F212, CS F213",
        help="e.g. CS F211, MATH F211",
    )
    current_raw = st.text_area(
        "Currently enrolled courses (comma-separated)",
        value="",
    )
    minor = st.text_input("Minor (optional)", value="")
    interests_raw = st.text_input("Interests (comma-separated)", value="AI, security")

    profile = StudentProfile(
        campus=campus,
        admission_year=int(admission_year),
        degree=degree,
        current_semester=int(current_semester),
        completed_courses=[c.strip().upper() for c in completed_raw.split(",") if c.strip()],
        current_courses=[c.strip().upper() for c in current_raw.split(",") if c.strip()],
        minor=minor or None,
        interests=[i.strip() for i in interests_raw.split(",") if i.strip()],
    )

# --- Main: requirement summary ---
remaining = compute_remaining_requirements(profile, rules, category_map)

st.subheader("Your Remaining Requirements")
if not remaining:
    st.warning(
        f"No programme rules found for '{degree}'. Check that programme_rules.json "
        f"has an entry matching this degree name."
    )
else:
    cols = st.columns(len(remaining))
    for col, (rule_type, info) in zip(cols, remaining.items()):
        with col:
            if info["required"] > 0:
                st.metric(rule_type, f"{info['remaining']} / {info['required']} left")
            else:
                st.metric(rule_type, "unverified count")
            st.caption(info["description"][:120] + ("…" if len(info["description"]) > 120 else ""))

st.divider()

# --- Query box ---
st.subheader("Ask for a Recommendation")
st.caption('Try: "Suggest an AI-related DEL with no midsem" or "I want an OPEL with no attendance requirement"')

query = st.text_input("Your question", value="", placeholder="Suggest a DEL related to AI with no midsem")

if query:
    with st.spinner("Finding eligible, matching courses…"):
        try:
            results = recommend(profile, query, courses, rules, category_map, top_k=5)
        except FileNotFoundError:
            results = None
            st.error(
                "Embedding index not found — run `python scripts/build_embeddings.py` first, "
                "then reload this page."
            )

    if results is not None:
        if not results:
            st.info(
                "No eligible courses matched every constraint in your query. "
                "Try loosening one condition (e.g. drop 'no midsem') to see what's available."
            )
        else:
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['course_code']}** — {r['title']}")
                    st.write(r["explanation"])

st.divider()
with st.expander("Debug: full eligible course list for your current profile"):
    from requirement_engine import get_eligible_courses
    eligible = get_eligible_courses(profile, courses, remaining, category_map)
    st.write(f"{len(eligible)} eligible courses")
    st.dataframe(
        [{"code": c.course_code, "title": c.title, "category": category_map.get(c.course_code, c.category)}
         for c in eligible],
        use_container_width=True,
    )