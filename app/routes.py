import os
import tempfile

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.schemas import QueryRequest, QueryResponse, Source, BooksResponse, TaskStatusResponse
from book_registry import load_registry, is_already_ingested
from config import settings
from data_processing import peek_book_metadata
from tasks import ingest_book

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

_CITATION_PROMPT = ChatPromptTemplate.from_template("""
You are a precise research assistant. Your sole job is to answer questions using ONLY the sources provided — do not use prior knowledge.

## Rules
- Cite every factual claim inline using [1], [2], etc., immediately after the claim.
- If multiple sources support the same claim, cite all of them: [1][2].
- If the sources do not contain enough information to answer the question, respond with: "The provided sources do not contain sufficient information to answer this question."
- Do not speculate, infer, or add information beyond what is explicitly stated in the sources.
- Keep your answer concise and direct.

## Question
{question}

## Sources
{sources}

## Answer
Provide your answer here, with inline citations. End with a "References" section listing each cited source by number.
""")


def _format_docs(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        info = f"[{i}] {meta.get('source', 'Unknown source')}"
        info += f", Author: {meta.get('author', 'Unknown author')}"
        info += f", chapter: {meta.get('chapter', 'Unknown')}"
        info += f", Chapter Title: {meta.get('chapter_title', 'Unknown')}"
        formatted.append(f"{info}\n{doc.page_content}")
    return "\n\n".join(formatted)


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/books", response_model=BooksResponse)
def list_books():
    return BooksResponse(books=load_registry())


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "vector_store_loaded": request.app.state.vector_store is not None}


@router.post("/ingest", response_model=TaskStatusResponse)
def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb", dir=settings.upload_dir) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    meta = peek_book_metadata(tmp_path)
    if not meta["title"]:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail="Could not find a TITLE: header in the file.")

    if is_already_ingested(meta["title"]):
        os.unlink(tmp_path)
        raise HTTPException(status_code=409, detail=f"'{meta['title']}' is already in the database.")

    task = ingest_book.delay(tmp_path, file.filename)
    return TaskStatusResponse(task_id=task.id, state="PENDING")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str):
    from celery_app import celery_app
    result = celery_app.AsyncResult(task_id)

    if result.state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            state="FAILURE",
            error=str(result.info),
        )

    if result.state == "SUCCESS":
        return TaskStatusResponse(
            task_id=task_id,
            state="SUCCESS",
            result=result.result,
        )

    meta = result.info if isinstance(result.info, dict) else {}
    return TaskStatusResponse(task_id=task_id, state=result.state, result=meta)


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    vector_store = request.app.state.vector_store
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(body.question)

    chain = _CITATION_PROMPT | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser()
    answer = chain.invoke({"question": body.question, "sources": _format_docs(retrieved_docs)})

    sources = [
        Source(
            index=i,
            source=doc.metadata.get("source", ""),
            author=doc.metadata.get("author", ""),
            chapter=doc.metadata.get("chapter", ""),
            chapter_title=doc.metadata.get("chapter_title", ""),
            content=doc.page_content,
        )
        for i, doc in enumerate(retrieved_docs, 1)
    ]

    return QueryResponse(answer=answer, sources=sources)
