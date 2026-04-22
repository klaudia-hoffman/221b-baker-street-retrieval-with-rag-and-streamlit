import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from app.schemas import BookEntry

REGISTRY_PATH = Path("book_registry.json")


def load_registry() -> List[BookEntry]:
    if not REGISTRY_PATH.exists():
        return []
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return [BookEntry(**entry) for entry in json.load(f)]


def is_already_ingested(title: str) -> bool:
    return any(e.title == title for e in load_registry())


def append_to_registry(title: str, author: str) -> BookEntry:
    entries = load_registry()
    entry = BookEntry(title=title, author=author, ingested_at=datetime.now(timezone.utc))
    entries.append(entry)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump([e.model_dump(mode="json") for e in entries], f, indent=2)
    return entry
