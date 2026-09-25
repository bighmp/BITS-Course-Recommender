"""
Batch-extracts structured Course + HandoutData records from every handout
PDF in data/raw/handouts/, using Gemini with native structured output.

free-tier Gemini RPM caps are tiny (5 RPM), and Google's servers 
sometimes return transient 503s
under their own load, independent of your rate limiting. This version
uses ONE global rate limiter that every call — first attempt AND
retries — passes through, so retries can never push you over your RPM
cap and trigger a self-inflicted 429 on top of Google's 503.

Bash:
    nohup python extract_handouts.py > extract.log 2>&1 &
    tail -f extract.log      # check progress
    # (script is resumable — safe to Ctrl-C or let the machine sleep
    #  and rerun later; already-succeeded files are skipped)

Usage:
    export GEMINI_API_KEY=...
    python extract_handouts.py

Requires: google-genai, pdfplumber, pydantic, tqdm
    pip install google-genai pdfplumber pydantic tqdm
"""

import asyncio
import json
import time
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types
from tqdm import tqdm

from schema import Course, ExtractedCourse

RAW_DIR = Path("data/raw/handouts")
OUT_PATH = Path("data/processed/courses.json")
MODEL = "gemini-3.8-flash"     

RPM_LIMIT = 5                  # set to your observed/confirmed free-tier cap
MIN_INTERVAL = 60.0 / RPM_LIMIT  # seconds between ANY two calls, incl. retries
MAX_RETRIES = 5                # 503s are transient Google-side overload 
RETRY_BACKOFF_BASE = 8         # seconds; grows per attempt (8, 16, 24, 32, 40)

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


class RateLimiter:
    """Ensures at least MIN_INTERVAL seconds between the start of any two
    calls, across the whole run — first attempts and retries share the
    same clock, so a flurry of retries can't blow past your RPM cap."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


limiter = RateLimiter(MIN_INTERVAL)


def pdf_to_text(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


async def extract_one(path: Path) -> dict | None:
    text = pdf_to_text(path)
    if not text.strip():
        return {"source_file": path.name, "error": "no extractable text (likely scanned image)"}

    for attempt in range(1, MAX_RETRIES + 1):
        await limiter.wait()  # every attempt, including retries, respects RPM
        try:
            resp = await client.aio.models.generate_content(
                model=MODEL,
                contents=EXTRACTION_PROMPT.format(text=text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedCourse,
                ),
            )
            extracted: ExtractedCourse = resp.parsed
            data = extracted.model_dump()
            data["source"] = {
                "source_file": path.name,
                "extraction_confidence": data.pop("extraction_confidence", "medium"),
            }
            course = Course.model_validate(data)
            return course.model_dump()
        except Exception as e:
            is_transient = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if attempt == MAX_RETRIES:
                return {"source_file": path.name, "error": str(e)}
            if is_transient:
                # extra backoff on top of the rate limiter, since a 503
                # means Google itself is asking everyone to back off
                await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)
            # non-transient errors (bad schema match, etc.) still get one
            # more try after the limiter's normal spacing, no extra wait


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
    print(f"Rate-limited to {RPM_LIMIT} RPM -> roughly {len(todo) * MIN_INTERVAL / 60:.0f} min minimum, more with retries.")

    results = list(already_done.values())

    # sequential by design: RPM_LIMIT=5 means concurrency is no use
    for path in tqdm(todo):
        r = await extract_one(path)
        if r:
            results.append(r)
            OUT_PATH.write_text(json.dumps(results, indent=2))  # checkpoint every file

    failed = [r for r in results if "error" in r]
    print(f"Done. {len(results) - len(failed)} succeeded, {len(failed)} failed/need review.")
    if failed:
        print("Failed files (rerun the script to retry just these):", [f["source_file"] for f in failed])


if __name__ == "__main__":
    asyncio.run(main())