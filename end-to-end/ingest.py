"""
Ingestion: read documents from data/, clean, chunk (with page numbers), embed, and write
to a local Chroma store.

Every run REBUILDS the collection from scratch. That is deliberate: with random IDs a
re-run silently doubles the corpus, and with stable IDs alone an edited or deleted
document would leave stale chunks behind.

    python ingest.py
    python ingest.py --data-dir papers --size 1000
"""

import argparse
from pathlib import Path

from langchain_chroma import Chroma

import config
import models
from chunking import make_chunks

SUPPORTED = {".pdf", ".txt", ".md"}


def load_pages(path: Path):
    """Return ([(page_number, text), ...], kind)."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = []
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                print(f"  warning: {path.name} page {number} has no extractable text (scanned?)")
            pages.append((number, text))
        return pages, "pdf"
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))], "text"


def build_chunks(data_dir, size=config.CHUNK_SIZE, overlap=None):
    overlap = size // 5 if overlap is None else overlap
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    chunks = []
    for path in files:
        pages, kind = load_pages(path)
        chunks += make_chunks(pages, path.relative_to(data_dir).as_posix(), size, overlap, kind)
    return chunks


def ingest(data_dir=config.DATA_DIR, persist_dir=config.PERSIST_DIR,
           size=config.CHUNK_SIZE, overlap=None, embeddings=None):
    chunks = build_chunks(data_dir, size, overlap)
    if not chunks:
        raise SystemExit(f"No readable .pdf/.txt/.md files found in {data_dir}/")
    ids = [c.metadata["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate chunk ids; chunk_id must be unique per chunk.")

    emb = embeddings or models.get_embeddings()
    store = dict(collection_name=config.COLLECTION_NAME, persist_directory=str(persist_dir))
    Chroma(embedding_function=emb, **store).delete_collection()      # rebuild from scratch
    Chroma.from_documents(chunks, emb, ids=ids, **store)
    return len(chunks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--persist-dir", default=str(config.PERSIST_DIR))
    ap.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    args = ap.parse_args(argv)

    print(f"Ingesting {args.data_dir}/ (chunk size {args.size}) ...")
    n = ingest(args.data_dir, args.persist_dir, args.size)
    print(f"Done: {n} chunks written to {args.persist_dir}/")


if __name__ == "__main__":
    main()
