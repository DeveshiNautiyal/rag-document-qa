"""Self-check for upload guardrails: chunk cap, upload size cap, DEV_MODE bypass.

Run: MISTRAL_API_KEY=dummy uv run python tests/test_limits.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("MISTRAL_API_KEY", "dummy")

from fastapi import HTTPException  # noqa: E402

import simple_rag.api as api  # noqa: E402
import simple_rag.rag as rag  # noqa: E402


def test_chunk_limit_raises_when_exceeded():
    rag.DEV_MODE = False
    try:
        rag._enforce_chunk_limit(["x"] * (rag.MAX_CHUNKS + 1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_chunk_limit_passes_within_bound():
    rag.DEV_MODE = False
    rag._enforce_chunk_limit(["x"] * rag.MAX_CHUNKS)  # should not raise


def test_chunk_limit_bypassed_in_dev_mode():
    rag.DEV_MODE = True
    try:
        rag._enforce_chunk_limit(["x"] * (rag.MAX_CHUNKS + 1))  # should not raise
    finally:
        rag.DEV_MODE = False


def test_upload_size_rejects_oversized_file():
    api.DEV_MODE = False
    try:
        api._enforce_upload_size(b"x" * (api.MAX_UPLOAD_BYTES + 1))
        raise AssertionError("expected HTTPException")
    except HTTPException as e:
        assert e.status_code == 413


def test_upload_size_passes_within_bound():
    api.DEV_MODE = False
    api._enforce_upload_size(b"x" * api.MAX_UPLOAD_BYTES)  # should not raise


def test_upload_size_bypassed_in_dev_mode():
    api.DEV_MODE = True
    try:
        api._enforce_upload_size(b"x" * (api.MAX_UPLOAD_BYTES + 1))  # should not raise
    finally:
        api.DEV_MODE = False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
