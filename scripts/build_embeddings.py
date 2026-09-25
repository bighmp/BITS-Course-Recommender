"""
Shared data schema for the course recommender's structured dataset.
Every extraction script writes records that conform to these shapes.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class SourceMeta(BaseModel):
    source_file: str
    extraction_confidence: Literal["high", "medium", "low", "needs_verification"] = "high"


class HandoutData(BaseModel):
    attendance_policy: Optional[str] = None
    has_midsem: Optional[bool] = None
    has_compre: Optional[bool] = None
    evaluation_components: list[str] = Field(default_factory=list)
    makeup_policy: Optional[str] = None
    instructor: Optional[str] = None
    topics: list[str] = Field(default_factory=list)  # used for embedding/semantic search


class Course(BaseModel):
    course_code: str          # normalized e.g. "BIO F311"
    title: str
    department: str
    units: Optional[float] = None
    category: Optional[str] = None   # CDC / DEL / HUEL / OPEL / CE etc.
    topics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    handout: Optional[HandoutData] = None
    source: SourceMeta


class ExtractedCourse(BaseModel):
    """Same shape as Course but WITHOUT `source` — this is what we hand to
    Gemini as response_schema, since the model shouldn't be inventing
    source metadata. extract_handouts.py attaches `source` after the call
    to build a full Course."""
    course_code: str
    title: str
    department: str
    units: Optional[float] = None
    category: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    handout: Optional[HandoutData] = None
    extraction_confidence: Literal["high", "medium", "low", "needs_verification"] = "high"


class ProgrammeRule(BaseModel):
    programme: str             # e.g. "B.E. Biological Sciences"
    rule_type: Literal["CDC", "DEL", "HUEL", "OPEL", "other"]
    description: str
    min_required: Optional[int] = None
    batch_scope: Optional[str] = None   # which batches this applies to, if stated
    source: SourceMeta


class TimetableEntry(BaseModel):
    course_code: str
    section: str
    instructor: Optional[str] = None
    days: Optional[str] = None
    hours: Optional[str] = None
    room: Optional[str] = None
    midsem_slot: Optional[str] = None
    compre_slot: Optional[str] = None


class StudentProfile(BaseModel):
    campus: str
    admission_year: int
    degree: str
    current_semester: int
    completed_courses: list[str] = Field(default_factory=list)
    current_courses: list[str] = Field(default_factory=list)
    minor: Optional[str] = None
    interests: list[str] = Field(default_factory=list)