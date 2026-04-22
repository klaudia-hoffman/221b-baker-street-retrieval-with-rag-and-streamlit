from datetime import datetime

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    index: int
    source: str
    author: str
    chapter: str
    chapter_title: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    message: str
    chapters_parsed: int
    chunks_added: int


class BookEntry(BaseModel):
    title: str
    author: str
    ingested_at: datetime


class BooksResponse(BaseModel):
    books: list[BookEntry]


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict | None = None
    error: str | None = None