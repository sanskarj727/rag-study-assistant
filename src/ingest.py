"""
ingest.py
---------
This module handles the very first stage of a RAG (Retrieval-Augmented
Generation) pipeline: getting raw documents into a form we can work with.

Two jobs happen here:
1. LOADING  - read raw text out of files (.txt and .pdf for now)
2. CHUNKING - break that text into smaller, overlapping pieces ("chunks")

Why chunk at all?
------------------
Large Language Models can only "see" a limited amount of text at once (their
context window). If we handed the model an entire 200-page textbook every
time we asked one question, it would be slow, expensive, and often *less*
accurate - models tend to get "lost" in very long inputs and miss the
relevant part.

Instead, RAG systems break documents into small chunks (e.g. a few hundred
words), turn each chunk into a vector in a later step, and only retrieve the
handful of chunks that are actually relevant to a given question. The model
then only has to read those few chunks, not the whole document.

Why overlap chunks?
---------------------
With zero overlap, we risk cutting an idea exactly in half across two chunks
and losing the context needed to answer a question about it. A small overlap
(e.g. the last 50 words of one chunk repeated as the first 50 of the next)
means content near a boundary is likely to appear in full in at least one
chunk.
"""

from pathlib import Path
from typing import List

from pypdf import PdfReader


def load_text_file(path: str) -> str:
    """Read a plain text (.txt) file and return its contents as one string."""
    return Path(path).read_text(encoding="utf-8")


def load_pdf_file(path: str) -> str:
    """
    Read a PDF file and return its contents as one string.

    A PDF doesn't store text as one clean stream - a reader library has to
    reconstruct it page by page. We loop over every page, pull the text out
    of each one, and join everything together with newlines.
    """
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def load_document(path: str) -> str:
    """
    Dispatch to the right loader based on the file extension.

    This is a small "factory" pattern: one entry point that hides the detail
    of *which* loader gets used. As we support more file types later (e.g.
    .docx), we only need to add one more `elif` here - nothing that calls
    load_document() needs to change.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    elif suffix == ".pdf":
        return load_pdf_file(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. Only .txt and .pdf are supported right now."
        )


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split `text` into overlapping chunks of `chunk_size` words.

    Parameters
    ----------
    text       : the full document text
    chunk_size : how many words go in each chunk
    overlap    : how many words from the end of one chunk are repeated at
                 the start of the next chunk

    This is a WORD-based, not sentence-based, chunker: it splits purely on
    whitespace and doesn't know or care about sentence boundaries. That's a
    deliberate simplification - it's easy to reason about and is a common
    real-world baseline. A natural next upgrade (worth trying yourself) is a
    sentence-aware chunker that never splits mid-sentence.
    """
    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap, or the window never moves forward "
            "and this function will loop forever."
        )

    words = text.split()  # naive whitespace split: "a  b\nc" -> ["a", "b", "c"]
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move the window forward by (chunk_size - overlap) words, not by
        # chunk_size. That's what makes the last `overlap` words of this
        # chunk reappear at the start of the next one.
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    # A tiny manual test you can run directly with: python src/ingest.py
    # (run it from the project root, so the relative path below resolves)
    sample_path = Path(__file__).parent.parent / "data" / "sample.txt"
    text = load_document(str(sample_path))

    # Small chunk_size here so you can actually see multiple chunks form
    # from a short sample document.
    chunks = chunk_text(text, chunk_size=60, overlap=15)

    print(f"Loaded {len(text.split())} words total from {sample_path.name}")
    print(f"Split into {len(chunks)} chunks (chunk_size=60, overlap=15)\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ({len(chunk.split())} words) ---")
        print(chunk)
        print()
