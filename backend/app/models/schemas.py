from pydantic import BaseModel
from typing import List, Optional


class Chunk(BaseModel):
    id: str
    content: str
    metadata: dict
    framework: str
    version: str
    url: str
    section: str


class Question(BaseModel):
    query: str
    framework: Optional[str] = None
    top_k: int = 5


class Source(BaseModel):
    url: str
    section: str
    content: str
    score: float


class Answer(BaseModel):
    answer: str
    sources: List[Source]
    query: str
