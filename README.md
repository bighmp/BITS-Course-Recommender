# BITS Academic Course Recommender

A dashboard that helps a BITS student choose courses for a semester. It first works out what the student **still needs** and is **eligible** to take using deterministic academic rules, and only then ranks eligible courses against the student's natural-language preferences.

```text
Student Profile + Query
  -> Remaining CDC / DEL / HUEL / OPEL      (requirement_engine.py, deterministic)
  -> Eligible course set                    (requirement_engine.py, deterministic)
  -> Constraint filters + preference match  (recommend.py)
  -> Explained recommendations              (template-based explanations)
````

The core design principle is that **academic eligibility is deterministic**. Similarity search and Gemini are never used to decide whether a course is academically valid.

## Setup

Requires Python 3.10+.

```powershell
git clone <this-repo>
cd BITS-Course-Recommender

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

The default recommendation mode is local and does not require an API key.

Gemini is optional. To enable Gemini Enhanced mode:

```powershell
$env:GEMINI_API_KEY="<your-api-key>"
```

## Data Layout

```text
data/
  raw/
    handouts/               # course handout PDFs
    bulletin-210-313.pdf    # BITS bulletin excerpt
  processed/
    courses.json
    programme_rules.json
    course_category_map.json
  embeddings/
    # locally generated semantic index
```

The raw academic documents are used as the source material for the structured dataset.

## Build the Dataset

Run everything from the repository root.

```powershell
# 1. Handouts -> structured course data
python scripts/extract_handouts_free.py

# 2. Bulletin -> programme requirements
python scripts/extract_rules.py data/raw/bulletin-210-313.pdf

# 3. Bulletin -> programme-aware course categories
python scripts/extract_category_map.py data/raw/bulletin-210-313.pdf

# 4. Build the local semantic search index
python scripts/build_embeddings.py
```

The handout extraction pipeline uses deterministic parsing and reports records that could not be parsed instead of silently treating them as valid courses.

The category map is **programme-aware**: CDC and DEL mappings are associated with the relevant programme, while HUEL mappings are stored globally.

## Run the Dashboard

```powershell
streamlit run scripts/app.py
```

Open the Streamlit dashboard and fill in the student profile:

* Campus
* Admission year
* Degree / programme
* Current semester
* Completed courses
* Current courses
* Minor
* Interests

Then enter a natural-language request.

Example queries:

```text
Suggest a DEL related to AI
Suggest HUELs
Suggest an AI-related DEL with no midsem
Suggest a HUEL with project-based evaluation
I want an OPEL with no attendance requirement
```

The first query may take longer because the local embedding model is loaded/downloaded. Later queries reuse the loaded model.

## Recommendation Modes

### Local

The default mode uses a deterministic keyword-based intent parser.

It can identify:

* CDC
* DEL
* HUEL
* OPEL
* academic topics such as AI, machine learning, NLP, security, cryptography, data science, etc.
* no-attendance requests
* no-midsem requests
* no-compre requests
* lenient makeup requests
* project-based evaluation requests

Local mode requires **no API calls**.

### Gemini Enhanced

The optional Gemini mode is used for richer natural-language intent extraction.

Gemini receives the student's query and extracts preferences such as:

* requested course category
* academic topics
* course-property constraints

It is explicitly instructed **not to determine academic eligibility or invent BITS rules**.

If Gemini fails or is unavailable, the recommender automatically falls back to the local parser.

## Academic Eligibility

`requirement_engine.py` is responsible for deterministic academic filtering.

For a course to reach the recommendation stage, the engine checks:

1. The course has a known category.
2. The category has a verified requirement.
3. The student still has a requirement remaining in that category.
4. The course has not already been completed.
5. Explicitly represented prerequisites are satisfied.

Unknown categories are **not** assumed to be eligible.

Programme-specific category mappings are normalized so formatting differences such as:

```text
B.E. Computer Science
B. E. Computer Science
```

do not affect programme matching.

Example category lookups for Computer Science include:

```text
CS F211  -> CDC
CS F407  -> DEL
HSS F222 -> HUEL
```

## Course Recommendation Pipeline

```text
1. Student profile
        |
        v
2. Query intent extraction
   Local parser / Gemini
        |
        v
3. Remaining requirements
        |
        v
4. Deterministic eligible courses
        |
        v
5. Category + course-property filters
        |
        v
6. Semantic similarity ranking
        |
        v
7. Top recommendations + explanation
```

This separation ensures that a course cannot become academically eligible simply because it is semantically similar to the student's query.

## Repository Guide

| File                               | Role                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `scripts/schema.py`                | Pydantic models for courses, programme rules and student profiles             |
| `scripts/extract_handouts_free.py` | Deterministic handout extraction                                              |
| `scripts/extract_rules.py`         | Extracts programme requirements from the bulletin                             |
| `scripts/extract_category_map.py`  | Builds programme-aware CDC/DEL and global HUEL mappings                       |
| `scripts/build_embeddings.py`      | Builds the local sentence-transformer semantic index                          |
| `scripts/requirement_engine.py`    | Requirement calculation, category lookup, prerequisite and eligibility checks |
| `scripts/recommend.py`             | Intent extraction, constraint filtering, semantic ranking and explanations    |
| `scripts/app.py`                   | Streamlit dashboard                                                           |
| `scripts/extract_handouts.py`      | Optional Gemini-based handout extraction                                      |

## Data / Extraction Notes

The current processed dataset contains **501 valid course records**. Some source handouts cannot be parsed reliably and are skipped by the extraction pipeline.

The requirement engine does not silently convert missing academic information into a valid rule. When a required property is not available, the system avoids making an unsupported eligibility decision.

## Known Limitations

* Some course handouts contain incomplete or difficult-to-parse information and are skipped during extraction.
* Course properties such as attendance, midsem, compre and makeup policy can only be checked when they are represented in the extracted handout.
* OPEL courses are not inferred from an unknown course category.
* Timetable clash checking is not implemented.
* Dual-degree-specific requirement handling is not implemented.
* The local intent parser is keyword-based and therefore does not understand every possible natural-language phrasing.
* Gemini Enhanced mode depends on Gemini API availability and quota.
* Semantic ranking is based on the locally generated course embeddings.
* Recommendation explanations are template-based rather than LLM-generated.



