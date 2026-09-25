"""
Batch-extracts structured Course + HandoutData records from every handout
PDF in data/raw/handouts/, using Gemini 2.5 Flash with native structured
output (response_schema=ExtractedCourse) — no manual JSON parsing or
fence-stripping needed.

Built for ~500+ files: bounded concurrency, retries, and checkpointing so
a crash or rate-limit doesn't cost you the run.

Gemini 2.5 Flash has a 1M-token context window.

Usage:
    export GEMINI_API_KEY=...
    python extract_handouts.py

Requires: google-genai, pdfplumber, pydantic, tqdm
    pip install google-genai pdfplumber pydantic tqdm
"""

import asyncio
import json
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

from schema import Course, ExtractedCourse

RAW_DIR = Path("data/raw/handouts")
OUT_PATH = Path("data/processed/courses.json")
CONCURRENCY = 5          
MAX_RETRIES = 3
MODEL = "gemini-3.8-flash"       # swap to "gemini-2.5-flash-lite" for even cheaper/faster

client = genai.Client()  # reads GEMINI_API_KEY from env

EXTRACTION_PROMPT = """You will be given the raw text of a BITS Pilani course handout PDF.
Extract course details and handout policy details as structured data.

Read the ENTIRE document carefully — evaluation scheme, grading policy, and
make-up policy are usually in numbered sections near the end (e.g. sections
6-9) and are just as important as the course description at the top.

If a field cannot be found in the text, leave it null (or an empty list for
list fields) — never invent a value. Set extraction_confidence to
"needs_verification" if the course code or title is ambiguous.

HANDOUT TEXT:
{text}
"""


def pdf_to_text(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


async def extract_one(path: Path, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        text = pdf_to_text(path)
        if not text.strip():
            return {"source_file": path.name, "error": "no extractable text (likely scanned image)"}
        # Force a 12-second wait to respect the 5 RPM limit
        await asyncio.sleep(12)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=EXTRACTION_PROMPT.format(text=text),  # full text, no truncation
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractedCourse,
                    ),
                )
                extracted: ExtractedCourse = resp.parsed  # already validated against the schema
                data = extracted.model_dump()
                data["source"] = {
                    "source_file": path.name,
                    "extraction_confidence": data.pop("extraction_confidence", "medium"),
                }
                course = Course.model_validate(data)
                return course.model_dump()
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"source_file": path.name, "error": str(e)}
                await asyncio.sleep(2 * attempt)


async def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    already_done = {}
    if OUT_PATH.exists():
        already_done = {r["source"]["source_file"]: r for r in json.loads(OUT_PATH.read_text()) if "source" in r}
        print(f"Resuming: {len(already_done)} files already extracted.")

    all_pdfs = sorted(RAW_DIR.glob("*.pdf"))
    todo = [p for p in all_pdfs if p.name not in already_done]
    print(f"{len(all_pdfs)} total handouts, {len(todo)} left to extract.")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = list(already_done.values())

    tasks = [extract_one(p, sem) for p in todo]
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        r = await coro
        if r:
            results.append(r)
            # checkpoint every result so a crash never loses progress
            OUT_PATH.write_text(json.dumps(results, indent=2))

    failed = [r for r in results if "error" in r]
    print(f"Done. {len(results) - len(failed)} succeeded, {len(failed)} failed/need review.")
    if failed:
        print("Failed files:", [f["source_file"] for f in failed])


if __name__ == "__main__":
    asyncio.run(main())