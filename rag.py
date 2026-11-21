"""
Core RAG (Retrieval-Augmented Generation) logic.

This module contains all the reusable pieces: PDF extraction, chunking,
embedding, retrieval, and answer generation. It is interface-agnostic —
both the CLI and the FastAPI app import from here.
"""

import os
import re
import time
import uuid

import fitz  # PyMuPDF
import numpy as np
from fastembed import TextEmbedding
from openai import APIStatusError, OpenAI

# --- Config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set the GROQ_API_KEY environment variable.")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # local, free — ONNX via fastembed, no torch
CHAT_MODEL = "openai/gpt-oss-120b"  # Groq, free tier
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_RETRIES = 3
SESSION_TTL_SECONDS = 3600  # 1 hour of inactivity before a session is purged
MAX_CHUNKS = 2000  # caps chunks per document (~2M chars, ~400-500 pages — a book)
DEV_MODE = os.environ.get("ENVIRONMENT", "production").lower() == "development"

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
_embedder = TextEmbedding(model_name=EMBED_MODEL)

# Session-scoped document store:
#   session_documents[session_id][doc_id] = { "name", "chunks", "embeddings", "num_chunks" }
session_documents: dict[str, dict[str, dict]] = {}
# Tracks the last time each session was active (unix timestamp)
session_last_seen: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RETRYABLE_CODES = {429, 500, 503, 504}  # rate-limited or transiently unavailable


def _retry_with_backoff(fn, *args, **kwargs):
    """Retry a function call with exponential backoff on rate-limit or
    transient-unavailability errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIStatusError as e:
            if e.status_code in _RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                wait_time = 2**attempt * 15  # 15s, 30s, 60s
                print(
                    f"  {e.status_code}. Waiting {wait_time}s before retry "
                    f"({attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait_time)
            else:
                raise


def _session_docs(session_id: str) -> dict[str, dict]:
    """Return (and lazily create) the document store for *session_id*."""
    session_last_seen[session_id] = time.time()
    if session_id not in session_documents:
        session_documents[session_id] = {}
    return session_documents[session_id]


def cleanup_expired_sessions() -> int:
    """Purge sessions that have been idle longer than SESSION_TTL_SECONDS.

    Returns the number of sessions removed.
    """
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, ts in session_last_seen.items() if ts < cutoff]
    for sid in expired:
        session_documents.pop(sid, None)
        session_last_seen.pop(sid, None)
    return len(expired)


# ---------------------------------------------------------------------------
# 1. Extract text from a document (PDF or EPUB; file path or raw bytes)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".epub"}


def extract_text_from_path(doc_path: str) -> str:
    doc = fitz.open(doc_path)  # filetype inferred from the path's extension
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_bytes(file_bytes: bytes, filename: str = "upload.pdf") -> str:
    doc = fitz.open(stream=file_bytes, filetype=filename)  # inferred from filename
    return "\n".join(page.get_text() for page in doc)


# ---------------------------------------------------------------------------
# 2. Split text into overlapping chunks on sentence boundaries
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex that handles common
    abbreviations and decimal numbers reasonably well."""
    # Split on period/question-mark/exclamation followed by whitespace and
    # an uppercase letter, or on newlines that look like paragraph breaks.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Further split on double-newlines (paragraph boundaries)
    result: list[str] = []
    for s in sentences:
        parts = re.split(r"\n{2,}", s)
        result.extend(p.strip() for p in parts if p.strip())
    return result


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into chunks of approximately *size* characters, breaking
    on sentence boundaries so that no chunk starts or ends mid-word.
    Consecutive chunks share roughly *overlap* characters of context."""

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence is longer than the target size, add it as
        # its own chunk rather than dropping it.
        if sentence_len > size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
            continue

        # Would adding this sentence exceed the target size?
        if current_length + sentence_len + 1 > size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Build overlap: walk backwards through current_chunk until we
            # have approximately *overlap* characters.
            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s) + 1

            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for the joining space

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# 3. Embed texts locally (fastembed, ONNX runtime, no API call)
# ---------------------------------------------------------------------------


def embed(texts: list[str], on_progress=None) -> list[list[float]]:
    """Embed *texts*. If *on_progress* is given, it's called as
    on_progress(done, total) after each vector — fastembed yields lazily,
    so a whole-book upload can report real progress instead of blocking
    silently for minutes."""
    embeddings = []
    for i, vector in enumerate(_embedder.embed(texts), start=1):
        embeddings.append(vector)
        if on_progress:
            on_progress(i, len(texts))
    return embeddings


# ---------------------------------------------------------------------------
# 4. Retrieve top-k relevant chunks (cosine similarity)
# ---------------------------------------------------------------------------


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    top_k: int = TOP_K,
) -> list[str]:
    query_embedding = embed([query])[0]
    scores = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# 5. Generate an answer with retrieved context
# ---------------------------------------------------------------------------


def ask(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful research assistant. Answer the user's question
based on the context passages provided below. Use the information in the context
to give a thorough, accurate answer. If the context contains relevant details,
synthesize them into a clear response. Only say "I don't know" if the context
truly contains no relevant information at all.

Context:
{context}

Question: {query}

Answer:"""
    response = _retry_with_backoff(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 6. High-level document operations (used by both CLI and API)
# ---------------------------------------------------------------------------

# ── CLI helpers (no session) ─────────────────────────────────────────────────
# The CLI uses a single shared session constant so existing main.py / eval.py
# code keeps working without modification.
_CLI_SESSION = "cli"


def ingest_pdf_path(pdf_path: str) -> str:
    """Load a PDF from disk, chunk & embed it, return a document id (CLI use)."""
    text = extract_text_from_path(pdf_path)
    return _ingest(text, name=os.path.basename(pdf_path), session_id=_CLI_SESSION)


# ── API helpers (session-scoped) ─────────────────────────────────────────────


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    session_id: str = _CLI_SESSION,
    on_progress=None,
) -> str:
    """Load a PDF from raw bytes, chunk & embed it, return a document id."""
    text = extract_text_from_bytes(pdf_bytes, filename)
    return _ingest(text, name=filename, session_id=session_id, on_progress=on_progress)


def _enforce_chunk_limit(chunks: list[str]) -> None:
    """Reject documents that would need an unreasonable number of embedding
    calls (cost/DoS guard for the public-facing API)."""
    if DEV_MODE:
        return
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"Document too large: {len(chunks)} chunks exceeds the "
            f"{MAX_CHUNKS}-chunk limit."
        )


def _ingest(text: str, name: str, session_id: str, on_progress=None) -> str:
    chunks = chunk_text(text)
    _enforce_chunk_limit(chunks)
    embeddings = embed(chunks, on_progress=on_progress)
    doc_id = uuid.uuid4().hex[:12]
    _session_docs(session_id)[doc_id] = {
        "name": name,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "embeddings": embeddings,
    }
    return doc_id


def query_document(doc_id: str, question: str, session_id: str = _CLI_SESSION) -> dict:
    """Retrieve context and generate an answer for *question* against *doc_id*."""
    doc = _session_docs(session_id).get(doc_id)
    if doc is None:
        raise KeyError(f"Document '{doc_id}' not found.")

    relevant = retrieve(question, doc["chunks"], doc["embeddings"])
    answer = ask(question, relevant)
    return {
        "answer": answer,
        "context_chunks": relevant,
    }


def delete_document(doc_id: str, session_id: str = _CLI_SESSION) -> None:
    """Remove a document from the session's in-memory store."""
    docs = _session_docs(session_id)
    if doc_id not in docs:
        raise KeyError(f"Document '{doc_id}' not found.")
    del docs[doc_id]


def list_documents(session_id: str = _CLI_SESSION) -> list[dict]:
    """Return metadata for every document loaded in this session."""
    return [
        {"doc_id": doc_id, "name": doc["name"], "num_chunks": doc["num_chunks"]}
        for doc_id, doc in _session_docs(session_id).items()
    ]
