import os

from celery_app import celery_app
from book_registry import append_to_registry, is_already_ingested
from data_processing import (
    load_book_with_metadata_by_chapter,
    apply_semantic_chunking,
    load_vector_store,
)


@celery_app.task(bind=True)
def ingest_book(self, file_path: str, original_filename: str) -> dict:
    try:
        self.update_state(state="STARTED", meta={"step": "parsing chapters"})
        chapters = load_book_with_metadata_by_chapter(file_path)

        title = chapters[0].metadata.get("source", original_filename) if chapters else original_filename
        author = chapters[0].metadata.get("author", "Unknown") if chapters else "Unknown"

        if is_already_ingested(title):
            raise ValueError(f"'{title}' is already in the database.")

        self.update_state(state="STARTED", meta={"step": "semantic chunking"})
        chunks = apply_semantic_chunking(chapters)

        self.update_state(state="STARTED", meta={"step": "writing to vector store"})
        vector_store = load_vector_store()
        vector_store.add_documents(chunks)

        append_to_registry(title, author)

        return {
            "title": title,
            "author": author,
            "chapters_parsed": len(chapters),
            "chunks_added": len(chunks),
        }
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)
