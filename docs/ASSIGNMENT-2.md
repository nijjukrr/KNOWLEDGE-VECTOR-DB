# Assignment 2 - Intelligent University Knowledge Assistant

## Objective

Design and develop a Retrieval-Augmented Knowledge Assistant for university information that can process institutional documents, create a searchable repository, retrieve relevant information from multiple sources, answer questions using retrieved context, support document updates, and provide explainable source references.

## Requirement Mapping

| Assignment requirement | Implementation in this repository | Status |
| --- | --- | --- |
| Process institutional documents | `/api/upload` accepts TXT, MD, CSV, JSON and PDF; `rag_engine.py` extracts, chunks and tokenizes content | Completed |
| Create searchable knowledge repository | Chunks and retrieval metadata are persisted in `data/index.json` | Completed |
| Retrieve information from multiple sources | Queries are scored across all indexed chunks and top results are returned | Completed |
| Answer using retrieved context | `KnowledgeBase.answer()` composes the response only from retrieved document sentences | Completed |
| Support uploads and updates | Re-upload replaces a document; delete and rebuild endpoints are provided | Completed |
| Explainable responses with source references | Each answer returns document name, chunk number, score, confidence and excerpt | Completed |

## Architecture

### Ingestion / Indexing

```text
Institutional Document
        |
        v
Text Extraction
        |
        v
Chunking (120 words, 25-word overlap)
        |
        v
Tokenization + Local Index
        |
        v
data/index.json
```

### Query / Retrieval

```text
User Question
     |
     v
Tokenization
     |
     v
TF-IDF-style Chunk Scoring
     |
     v
Top-k Retrieved Chunks
     |
     v
Grounded Extractive Answer
     |
     v
Answer + Source References
```

## Main Components

- `app.py` - HTTP server, UI routes, upload/query/rebuild/delete API.
- `rag_engine.py` - extraction, chunking, indexing, retrieval, answer composition, source metadata.
- `templates/index.html` - interactive university knowledge assistant UI.
- `static/styles.css` - responsive styling.
- `sample_documents/` - academic policy, admissions FAQ, and library examples.
- `tests/test_rag_engine.py` - automated tests for indexing, updating, deletion and rebuilding.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Repository statistics |
| POST | `/api/upload` | Upload and index a document |
| POST | `/api/ask` | Ask a question using retrieved context |
| POST | `/api/rebuild` | Rebuild index from stored documents |
| DELETE | `/api/documents/{filename}` | Remove document and indexed chunks |

## How to Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

Upload the sample files in `sample_documents/`, then ask questions such as:

- What happens if attendance is below 75 percent?
- When does the central library close?
- What documents are required for admission?

## Testing

The automated tests verify:

1. An uploaded document is indexed and can answer a query.
2. Re-uploading the same file replaces outdated content.
3. A document can be removed.
4. The repository can be rebuilt from stored files.

Run:

```powershell
python -m unittest discover -s tests -v
```

## Important Implementation Note

The current project is a lightweight, dependency-free RAG-style prototype. Retrieval uses TF-IDF-style lexical scoring and answer generation is extractive.

A full production RAG implementation can later replace the retriever with dense semantic embeddings stored in ChromaDB, FAISS, Pinecone or Weaviate and use an LLM for context-grounded generation. This distinction is documented intentionally so the submission accurately describes the code.

## Conclusion

The project fulfills the six functional requirements of Assignment 2 by turning uploaded institutional files into a searchable local knowledge repository and returning document-grounded answers with transparent source references.
