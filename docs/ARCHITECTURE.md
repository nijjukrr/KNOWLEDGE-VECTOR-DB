# Architecture

Knowledge Vector DB is intentionally simple to run while still following the core RAG workflow.

## Flow

1. A user uploads an institutional document.
2. The server stores the original file in `data/documents`.
3. The engine extracts text, chunks it, tokenizes it, and stores an index in `data/index.json`.
4. A user asks a question.
5. The engine scores chunks with TF-IDF style retrieval.
6. The answer is composed from the strongest matching sentences.
7. The UI shows the answer and source references.

## Components

- `app.py` exposes the web UI and JSON API.
- `rag_engine.py` handles parsing, indexing, retrieval, answer synthesis, and repository statistics.
- `templates/index.html` provides the interactive dashboard.
- `static/styles.css` defines the responsive interface.

## Design Choice

The project avoids external services so it can run on any student laptop. A production deployment can replace the local retriever with embeddings, a vector database, OCR, authentication, and a hosted model.
