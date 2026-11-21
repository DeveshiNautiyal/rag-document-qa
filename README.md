# 🔍 Simple RAG

A minimal **Retrieval-Augmented Generation** system built with Python and
FastAPI. Upload a PDF or EPUB, ask questions about it, and get accurate
answers grounded in the document's content — **$0 to run.**

This is meant to be bolted onto a bigger app, not the main event. If you're
adding "chat with this document" as one feature among many, you don't want an
API bill or a vector database to babysit for it. So:

- **Embeddings are free forever.** They run locally (ONNX via `fastembed`,
  no torch) — no API call, no key, no cost, no rate limit, no matter how much
  you use it.
- **Chat is free too**, on [Groq's](https://console.groq.com/keys) no-card
  free tier — fine for a side-feature's worth of traffic. Swap in a paid
  model later in one line ([`CHAT_MODEL`](#configuration)) if you outgrow it.
- **No vector database.** Embeddings live in memory as plain Python lists;
  retrieval is cosine similarity over a numpy array. No Pinecone account, no
  Docker container, nothing to provision — fine up to a few hundred pages.
- **~300 lines of core logic** (`rag.py`), interface-agnostic — everything
  else (API, CLI) just calls into it. See [Using this in your own
  app](#using-this-in-your-own-app) to skip the FastAPI/UI layer entirely and
  import the functions directly.

No vector databases, no heavyweight frameworks — just the core RAG ideas,
written to be read, forked, and dropped into whatever you're actually
building.

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that grounds LLM answers
in _your_ documents instead of relying on the model's training data alone.

```
┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│  Upload  │────▶│  Chunk   │────▶│  Embed    │────▶│  Store   │
│   PDF    │     │  Text    │     │  Chunks   │     │ Vectors  │
└──────────┘     └──────────┘     └───────────┘     └────┬─────┘
                                                         │
┌──────────┐     ┌──────────┐     ┌───────────┐         │        ┌──────────┐
│   Ask    │────▶│  Embed   │────▶│  Find Top │─────────┘        │          │
│ Question │     │  Query   │     │  Matches  │─────────────────▶│   Groq   │
└──────────┘     └──────────┘     └───────────┘ relevant chunks  │   LLM    │
                                                                 └────┬─────┘
                                                                      │
                                                                 ┌────▼─────┐
                                                                 │  Answer  │
                                                                 │ grounded │
                                                                 │ in docs  │
                                                                 └──────────┘
```

1. **Extract** — Pull text out of a PDF using PyMuPDF
2. **Chunk** — Split text into overlapping pieces on sentence boundaries
3. **Embed** — Convert each chunk into a vector locally using `BAAI/bge-small-en-v1.5`
   (fastembed, ONNX runtime, no torch, no API call)
4. **Retrieve** — Embed the question and find the most similar chunks using
   cosine similarity
5. **Generate** — Send the question + top-K chunks to `openai/gpt-oss-120b`
   on Groq, which answers from the document content

---

## Project Structure

```
simple_rag/
├── src/simple_rag/
│   ├── rag.py            # Core RAG logic — chunking, embedding, retrieval, generation
│   ├── api.py            # FastAPI REST API + UI routes
│   ├── main.py           # Interactive CLI
│   ├── eval.py           # LLM-as-a-judge evaluation harness
│   └── static/
│       ├── index.html    # Browser chat UI (HTML + JS)
│       └── style.css     # Stylesheet (editorial newspaper aesthetic)
├── pyproject.toml        # Project config & dependencies (uv)
└── README.md
```

| Module          | Purpose                                                                                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rag.py`        | The engine. PDF extraction, sentence-aware chunking, embedding, cosine-similarity retrieval, and answer generation. Both the CLI and API import from here.                   |
| `api.py`        | FastAPI app wrapping the RAG engine into REST endpoints. Also serves the browser UI at `/ui`. Includes auto-generated interactive docs at `/docs`.                          |
| `main.py`       | Minimal interactive CLI — load a PDF, ask questions in a loop.                                                                                                              |
| `eval.py`       | Automated evaluation harness: runs 10 test questions against "Attention is All You Need", uses the same chat model as a judge, and prints a scored report per difficulty.    |
| `static/`       | Single-page browser UI. No build step — plain HTML, CSS, and vanilla JS served directly by FastAPI.                                                                        |

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Groq API key](https://console.groq.com/keys) (free tier, no card
  required). Embeddings run locally, so no key is needed for those.

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd simple_rag

# Install dependencies
uv sync

# Set your API key
cp .env.example .env
# then edit .env and paste in your key
```

Every `uv run` command below picks up `.env` automatically if you add
`--env-file .env` (e.g. `uv run --env-file .env uvicorn simple_rag.api:app --reload`).
Alternatively just `export GROQ_API_KEY=...` in your shell.

---

## Usage

### 1. Browser UI

The easiest way to use Simple RAG. Start the server and open the UI in your
browser:

```bash
uv run uvicorn simple_rag.api:app --reload
# → http://localhost:8000/ui
```

From the UI you can:

- **Upload** a PDF via drag-and-drop or the file picker
- **Switch** between multiple loaded documents from the sidebar
- **Chat** with the document — answers include a collapsible *Sources* section
  showing the retrieved passages

### 2. CLI — Interactive Chat

Chat with a PDF directly in your terminal:

```bash
uv run python -m simple_rag.main path/to/document.pdf
```

```
Loading document: path/to/document.pdf
Ready! Ask questions (type 'quit' to exit)

You: What is the main contribution of this paper?
Assistant: The paper introduces the Transformer, a sequence-to-sequence model
based entirely on attention mechanisms, dispensing with recurrence and
convolutions entirely...

You: quit
```

### 3. API — REST Interface

Start the API server:

```bash
uv run uvicorn simple_rag.api:app --reload
# → http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

#### Endpoints

| Method   | Path                         | Description                                          |
| -------- | ---------------------------- | ----------------------------------------------------- |
| `GET`    | `/ui`                        | Browser chat UI                                       |
| `POST`   | `/documents`                 | Start ingesting a PDF/EPUB (chunk + embed), returns `upload_id` |
| `GET`    | `/documents/status/{upload_id}` | Poll ingestion progress/result for an upload       |
| `GET`    | `/documents`                 | List all loaded documents                             |
| `GET`    | `/documents/{doc_id}`        | Get metadata for a specific document                  |
| `DELETE` | `/documents/{doc_id}`        | Remove a document from memory                          |
| `POST`   | `/query`                     | Ask a question against a loaded document               |

Uploads run in the background and return immediately — poll the status
endpoint for progress. This matters because embedding runs locally on CPU:
a paper finishes in seconds, but a whole book can take a few minutes, and
polling is what tells you it's working instead of just hanging.

#### Example workflow

```bash
# Start an upload — returns immediately with an upload_id
curl -X POST http://localhost:8000/documents \
  -F "file=@path/to/paper.pdf"
# → { "upload_id": "9047de7d...", "status": "embedding" }

# Poll until it's ready (repeat every second or so)
curl http://localhost:8000/documents/status/9047de7d...
# → { "status": "embedding", "done": 30, "total": 52 }
# → { "status": "ready", "doc_id": "a1b2c3d4e5f6", "name": "paper.pdf", "num_chunks": 52 }

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "a1b2c3d4e5f6", "question": "What is the main contribution?"}'
# → { "answer": "...", "context_chunks": [...] }

# List loaded documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/a1b2c3d4e5f6
```

### 4. Evaluation — LLM-as-a-Judge

Run an automated evaluation against the bundled "Attention is All You Need" PDF:

```bash
uv run python -m simple_rag.eval
```

This runs 10 question/answer pairs at three difficulty levels (easy, medium,
hard) and uses the same chat model to judge correctness. Results are printed to the terminal
and saved to `eval_report.json`.

```
======================================================================
  RAG Evaluation — LLM-as-a-Judge
======================================================================

📄 Ingesting: src/simple_rag/arxiv.pdf

─── Question 1/10 [EASY] ───
  Q: Who are the authors of this paper?
  A: The authors are Ashish Vaswani, Noam Shazeer, Niki Parmar...
  → ✔ CORRECT

...

======================================================================
  RESULTS SUMMARY
======================================================================

  Overall Score:  90.0%
  Total:          10
  ✔ Correct:      8
  ◐ Partial:      2
  ✘ Incorrect:    0

      EASY:  100.0%  (5/5 correct, 0 partial)
    MEDIUM:   83.3%  (2/3 correct, 1 partial)
      HARD:   75.0%  (1/2 correct, 1 partial)
```

You can also point the evaluator at any other PDF:

```bash
uv run python -m simple_rag.eval path/to/your.pdf
```

---

## Using this in your own app

If RAG is a feature bolted onto something bigger, you have two options —
neither needs a database, an account beyond a free Groq key, or a monthly
bill.

**Option A — run it as a sidecar.** Point your main app's backend at this
API instead of embedding it. `POST /documents` to ingest, `POST /query` to
ask — see [Endpoints](#endpoints) above. Good if your app isn't Python, or
you'd rather keep it as a separate process.

**Option B — vendor `rag.py` directly.** It's interface-agnostic and has no
FastAPI/server dependency of its own — copy the file into your project and
call the functions straight from your existing backend:

```python
from simple_rag import rag

doc_id = rag.ingest_pdf_bytes(pdf_bytes, filename="manual.pdf")
result = rag.query_document(doc_id, "What's the return policy?")
print(result["answer"])
```

That's the entire integration surface. No server to run, no extra process —
it's four function calls (`ingest_pdf_path` / `ingest_pdf_bytes`,
`query_document`, `delete_document`, `list_documents`) sitting on top of a
plain in-memory dict, so there's nothing to migrate or tear down if you later
decide RAG deserves to be a first-class part of the app instead of a bolt-on.

---

## How it works — Key Design Decisions

### Sentence-aware chunking with overlap

Rather than splitting on fixed character boundaries (which can cut
mid-sentence), the chunker splits on sentence boundaries using a regex and then
assembles sentences into chunks of ~1024 characters. Each new chunk starts with
the last ~200 characters of the previous one to preserve context across chunk
boundaries.

### No vector database

Embeddings are stored in-memory as plain Python lists. Retrieval is a
brute-force cosine similarity scan across all chunks. This is fine for learning
and for documents up to a few hundred pages. For production scale, you'd swap
this out for something like Chroma, Qdrant, or pgvector.

### Rate-limit handling

Free-tier APIs can still be rate-limited. `rag.py` includes a `_retry_with_backoff`
helper that catches `429` errors and retries with exponential backoff (15s → 30s
→ 60s) before giving up.

### LLM-as-a-judge evaluation

Rather than writing per-question regex matchers, `eval.py` uses the chat model itself as
an impartial judge. The judge model is given the question, ground-truth answer,
and RAG answer, then returns a structured JSON verdict (`CORRECT` /
`PARTIALLY_CORRECT` / `INCORRECT`). This makes it easy to add new test cases
without writing custom scoring logic.

### Browser UI

The UI is a single-page app served directly by FastAPI — no Node.js, no build
step, no framework. `static/index.html` and `static/style.css` are read off
disk and served at `/ui`. The JS communicates with the same REST endpoints used
by the CLI.

### Abuse guardrails

The API is public and every query costs real Groq API usage (embeddings are
local and free), so three
things are capped by default:

- **Upload size** — rejected above 20MB (`MAX_UPLOAD_BYTES` in `api.py`).
  Enforced again at the edge via `request_body max_size` in the deployed
  Caddyfile, so oversized uploads never reach the container.
- **Chunks per document** — rejected above 2000 chunks (`MAX_CHUNKS` in
  `rag.py`, ~400-500 pages — enough for a full book).
- **Rate limiting** — `slowapi`, keyed by client IP: 5/min on `POST
  /documents`, 20/min on `POST /query`.

Set `ENVIRONMENT=development` to disable all three for local testing —
production (the default, unset) keeps them on.

---

## Configuration

Key constants in `rag.py` / `api.py` that you can tweak:

| Constant           | Default                | Effect                                                        |
| ------------------ | ----------------------- | -------------------------------------------------------------- |
| `EMBED_MODEL`       | `BAAI/bge-small-en-v1.5` | Local fastembed (ONNX) embedding model                         |
| `CHAT_MODEL`        | `openai/gpt-oss-120b` | Groq chat model for answer generation and eval judging       |
| `CHUNK_SIZE`        | `1024`                  | Target chunk size in characters                                |
| `CHUNK_OVERLAP`     | `200`                   | Characters of overlap between consecutive chunks               |
| `TOP_K`             | `5`                     | Number of chunks retrieved per query                           |
| `MAX_RETRIES`       | `3`                     | Retry attempts on rate-limit errors                            |
| `MAX_CHUNKS`        | `2000`                  | Max chunks per document before upload is rejected              |
| `MAX_UPLOAD_BYTES`  | `20MB`                  | Max upload size before it's rejected                           |
| `ENVIRONMENT`       | unset (production)      | Set to `development` to disable the above guardrails locally   |

---

## Dependencies

| Package            | Purpose                                      |
| ------------------ | --------------------------------------------- |
| `openai`            | API client (OpenAI-compatible; points at Groq for generation) |
| `fastembed`         | Local embedding model via ONNX runtime (no API call, no cost, no torch) |
| `pymupdf`          | PDF text extraction                          |
| `numpy`            | Cosine similarity computation                |
| `fastapi`          | REST API framework                           |
| `uvicorn`          | ASGI server                                  |
| `python-multipart` | File upload support for FastAPI              |
| `slowapi`          | Per-IP rate limiting on the upload/query endpoints |
