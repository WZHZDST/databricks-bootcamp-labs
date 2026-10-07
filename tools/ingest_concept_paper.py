#!/usr/bin/env python3
"""Ingest the concept-paper chapters (``docs/chapters``) into a persistent ChromaDB collection.

The markdown chapters carry YAML front matter (``title``, ``chapter``, ``tags``) which is
preserved as chunk metadata so queries can be filtered and the source located.

Usage
-----
    python tools/ingest_concept_paper.py                 # ingest with defaults
    python tools/ingest_concept_paper.py --reset         # wipe the collection first
    python tools/ingest_concept_paper.py --docs-dir docs/chapters --db-dir .chroma

The module is importable too::

    from tools.ingest_concept_paper import ingest, get_collection
    ingest()
    col = get_collection()
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

# Repo root is the parent of the ``tools`` directory that holds this file.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "chapters"
DEFAULT_DB_DIR = REPO_ROOT / ".chroma"
COLLECTION_NAME = "concept_paper"

# Chunking parameters (characters). Kept small so retrieval stays focused.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    """A single retrievable chunk with its metadata."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Return ``(front_matter_dict, body)`` for a markdown document."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    body = raw[match.end():]
    return (data if isinstance(data, dict) else {}), body


def _scalar_metadata(front_matter: dict, source: str) -> dict:
    """Flatten front matter into Chroma-compatible scalar metadata."""
    title = front_matter.get("title")
    chapter = front_matter.get("chapter")
    tags = front_matter.get("tags")
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(str(t) for t in tags)
    meta = {"source": source}
    if title is not None:
        meta["title"] = str(title)
    if chapter is not None:
        meta["chapter"] = str(chapter)
    if tags:
        meta["tags"] = str(tags)
    return meta


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to break on the last paragraph/sentence boundary within the window.
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                idx = window.rfind(sep)
                if idx > size // 2:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _nearest_heading(body: str, position: int) -> str | None:
    """Return the most recent markdown heading at or before ``position``."""
    heading = None
    for m in HEADING_RE.finditer(body):
        if m.start() > position:
            break
        heading = m.group(2).strip()
    return heading


def chunk_file(path: Path, docs_dir: Path) -> list[Chunk]:
    """Parse and chunk a single markdown file into :class:`Chunk` objects."""
    raw = path.read_text(encoding="utf-8")
    front_matter, body = parse_front_matter(raw)
    source = str(path.relative_to(docs_dir))
    base_meta = _scalar_metadata(front_matter, source)

    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(split_text(body)):
        # Locate the piece in the body to attach the nearest heading for context.
        found = body.find(piece[:40], cursor) if piece else -1
        if found != -1:
            cursor = found
        heading = _nearest_heading(body, cursor if found != -1 else 0)
        meta = dict(base_meta)
        meta["chunk_index"] = i
        if heading:
            meta["heading"] = heading
        digest = hashlib.sha1(f"{source}:{i}".encode("utf-8")).hexdigest()[:16]
        chunks.append(Chunk(id=f"{source}#{i}-{digest}", text=piece, metadata=meta))
    return chunks


def iter_markdown(docs_dir: Path) -> Iterable[Path]:
    """Yield markdown files under ``docs_dir`` in a stable order."""
    yield from sorted(p for p in docs_dir.rglob("*.md") if p.is_file())


def get_client(db_dir: Path | str = DEFAULT_DB_DIR):
    """Return a persistent ChromaDB client rooted at ``db_dir``."""
    import chromadb

    Path(db_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_dir))


def get_collection(db_dir: Path | str = DEFAULT_DB_DIR, name: str = COLLECTION_NAME):
    """Return (creating if needed) the concept-paper collection."""
    client = get_client(db_dir)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "description": "Concept paper chapters"},
    )


def ingest(
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    db_dir: Path | str = DEFAULT_DB_DIR,
    name: str = COLLECTION_NAME,
    reset: bool = False,
    batch_size: int = 128,
) -> int:
    """Ingest all chapters into ChromaDB and return the number of chunks stored."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    client = get_client(db_dir)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "description": "Concept paper chapters"},
    )

    all_chunks: list[Chunk] = []
    file_count = 0
    for path in iter_markdown(docs_dir):
        file_chunks = chunk_file(path, docs_dir)
        if file_chunks:
            file_count += 1
            all_chunks.extend(file_chunks)

    if not all_chunks:
        print(f"No chunks produced from {docs_dir}")
        return 0

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        print(f"  upserted {min(start + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    print(
        f"Ingested {len(all_chunks)} chunks from {file_count} files "
        f"into collection '{name}' at {Path(db_dir).resolve()}"
    )
    return len(all_chunks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest concept-paper chapters into ChromaDB.")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Directory with chapter markdown files.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Persistent ChromaDB directory.")
    parser.add_argument("--name", default=COLLECTION_NAME, help="Collection name.")
    parser.add_argument("--reset", action="store_true", help="Delete the collection before ingesting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ingest(docs_dir=args.docs_dir, db_dir=args.db_dir, name=args.name, reset=args.reset)


if __name__ == "__main__":
    main()
