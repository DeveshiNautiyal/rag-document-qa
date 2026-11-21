"""
Simple RAG (Retrieval-Augmented Generation) — CLI interface.

Usage:
    uv run python -m simple_rag.main [path/to/document.pdf]
"""

import sys

from simple_rag.rag import ingest_pdf_path, query_document


def main():
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    else:
        doc_path = input("Document path (PDF/EPUB): ").strip()

    print(f"Loading document: {doc_path}")
    doc_id = ingest_pdf_path(doc_path)
    print("Ready! Ask questions (type 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        result = query_document(doc_id, query)
        print(f"\nAssistant: {result['answer']}\n")


if __name__ == "__main__":
    main()
