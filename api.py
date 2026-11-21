"""
FastAPI interface for the Simple RAG system.

Run with:
    uv run uvicorn simple_rag.api:app --reload

Endpoints:
    GET    /                             Redirect to the browser UI
    POST   /documents                    Start ingesting a PDF/EPUB, returns upload_id
    GET    /documents/status/{upload_id} Poll ingestion progress/result for an upload
    GET    /documents                    List all loaded documents
    GET    /documents/{doc_id}           Get details for a specific document
    DELETE /documents/{doc_id}           Remove a document from memory
    POST   /query                        Ask a question against a loaded document
    GET    /ui                           Browser chat UI
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from simple_rag.rag import (
    DEV_MODE,
    SUPPORTED_EXTENSIONS,
    cleanup_expired_sessions,
    delete_document,
    ingest_pdf_bytes,
    list_documents,
    query_document,
)

_STATIC_DIR = Path(__file__).parent / "static"
_SESSION_COOKIE = "rag_session"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
# ponytail: Caddy also caps request_body size at the edge in infra's Caddyfile;
# this is a backstop for direct/non-Caddy access (e.g. local `docker compose up`).

# Tracks in-flight/finished ingestion jobs, keyed by upload_id:
#   {"status": "embedding"|"ready"|"error", "done": int, "total": int,
#    "doc_id": str|None, "name": str|None, "num_chunks": int|None, "error": str|None}
# ponytail: never purged — a long-running deployment accumulates one small
# dict per upload forever. Fine at hobby-project volume; add a TTL sweep
# (same pattern as cleanup_expired_sessions) if that ever matters.
_ingest_jobs: dict[str, dict] = {}

limiter = Limiter(key_func=get_remote_address, enabled=not DEV_MODE)


# ---------------------------------------------------------------------------
# Session cookie middleware
# ---------------------------------------------------------------------------


class SessionMiddleware(BaseHTTPMiddleware):
    """Assign a random session ID cookie to every visitor and expose it via
    ``request.state.session_id``."""

    async def dispatch(self, request: Request, call_next):
        session_id = request.cookies.get(_SESSION_COOKIE)
        if not session_id:
            session_id = uuid.uuid4().hex

        request.state.session_id = session_id
        response = await call_next(request)

        if not request.cookies.get(_SESSION_COOKIE):
            # httponly=False so the JS side can read it if needed in future;
            # samesite=lax is safe and avoids CSRF issues on GET requests.
            response.set_cookie(
                key=_SESSION_COOKIE,
                value=session_id,
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 7,  # 1 week
            )

        return response


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------


async def _cleanup_loop() -> None:
    """Every 10 minutes, purge sessions idle for more than SESSION_TTL_SECONDS."""
    while True:
        await asyncio.sleep(600)
        removed = cleanup_expired_sessions()
        if removed:
            print(f"[cleanup] Purged {removed} expired session(s).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


class PrettyJSONResponse(JSONResponse):
    """JSONResponse subclass that pretty-prints with 2-space indentation."""

    def render(self, content) -> bytes:
        import json

        return json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="Simple RAG API",
    description="Minimal RAG API — Groq generation, local embeddings.",
    version="0.1.0",
    default_response_class=PrettyJSONResponse,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SessionMiddleware)

# Mount the static directory (CSS, JS assets if ever added)
app.mount("/ui/static", StaticFiles(directory=_STATIC_DIR), name="ui-static")


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Redirect / -> /ui/."""
    return RedirectResponse(url="/ui/")


@app.get("/ui", include_in_schema=False)
async def ui_redirect():
    """Redirect /ui -> /ui/ so relative paths in index.html resolve correctly."""
    return RedirectResponse(url="/ui/")


@app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    """Serve the single-page RAG chat interface."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    doc_id: str
    question: str


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    context_chunks: list[str]


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    num_chunks: int


class IngestStartResponse(BaseModel):
    upload_id: str
    status: str


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _enforce_upload_size(pdf_bytes: bytes) -> None:
    """Reject uploads over MAX_UPLOAD_BYTES (cost/DoS guard)."""
    if DEV_MODE:
        return
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(pdf_bytes)} bytes exceeds the "
            f"{MAX_UPLOAD_BYTES}-byte limit.",
        )


def _run_ingest(
    upload_id: str, pdf_bytes: bytes, filename: str, session_id: str
) -> None:
    """Runs in a worker thread (via asyncio.to_thread) — embedding a whole
    book on CPU can take minutes, and this keeps that off the event loop so
    the server stays responsive to other requests while it runs."""

    def report(done: int, total: int) -> None:
        _ingest_jobs[upload_id].update(done=done, total=total)

    try:
        doc_id = ingest_pdf_bytes(
            pdf_bytes, filename, session_id=session_id, on_progress=report
        )
        docs = list_documents(session_id=session_id)
        doc = next(d for d in docs if d["doc_id"] == doc_id)
        _ingest_jobs[upload_id].update(
            status="ready",
            doc_id=doc_id,
            name=doc["name"],
            num_chunks=doc["num_chunks"],
        )
    except ValueError as e:
        _ingest_jobs[upload_id].update(status="error", error=str(e))
    except Exception as e:
        _ingest_jobs[upload_id].update(
            status="error", error=f"Failed to ingest PDF: {e}"
        )


@app.post("/documents", response_model=IngestStartResponse, status_code=202)
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload a PDF or EPUB file. Starts chunking + embedding in the
    background and returns immediately — poll GET /documents/status/{upload_id}
    for progress and the final doc_id."""
    session_id = request.state.session_id
    filename = file.filename or "upload.pdf"

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF and EPUB files are accepted. Got: {filename}",
        )

    pdf_bytes = await file.read()
    _enforce_upload_size(pdf_bytes)

    upload_id = uuid.uuid4().hex
    _ingest_jobs[upload_id] = {"status": "embedding", "done": 0, "total": 0}
    asyncio.create_task(
        asyncio.to_thread(_run_ingest, upload_id, pdf_bytes, filename, session_id)
    )
    return IngestStartResponse(upload_id=upload_id, status="embedding")


@app.get("/documents/status/{upload_id}")
async def get_ingest_status(upload_id: str):
    """Poll the progress/result of an upload started via POST /documents."""
    job = _ingest_jobs.get(upload_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found.")
    return job


@app.get("/documents", response_model=list[DocumentInfo])
async def get_documents(request: Request):
    """List all documents loaded in this session."""
    return list_documents(session_id=request.state.session_id)


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, request: Request):
    """Get metadata for a specific document."""
    docs = list_documents(session_id=request.state.session_id)
    doc = next((d for d in docs if d["doc_id"] == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(**doc)


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str, request: Request):
    """Remove a document from this session's in-memory store."""
    try:
        delete_document(doc_id, session_id=request.state.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DeleteResponse(doc_id=doc_id, message="Document deleted.")


@app.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query(req: QueryRequest, request: Request):
    """Ask a question against a previously uploaded document."""
    try:
        result = query_document(
            req.doc_id, req.question, session_id=request.state.session_id
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Document '{req.doc_id}' not found."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=result["answer"],
        context_chunks=result["context_chunks"],
    )
