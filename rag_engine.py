from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIMENSIONS = 384


@dataclass
class Chunk:
    id: str
    document_name: str
    chunk_number: int
    text: str
    tokens: list[str]
    page_number: int | None = None
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class EmbeddingEngine:
    """Creates normalized dense vectors, with an offline deterministic fallback.

    If sentence-transformers is installed and its model is available, the project uses
    MiniLM semantic embeddings. If not, it falls back to a local hashing vectorizer so
    the application still runs without external services.
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._model = None
        self._attempted_model_load = False
        self._backend_name = "hashing-local"

    @property
    def backend_name(self) -> str:
        self._load_model_if_available()
        return self._backend_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model_if_available()
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [[float(value) for value in vector] for vector in vectors]
        return [hashing_embedding(text) for text in texts]

    def _load_model_if_available(self) -> None:
        if self._attempted_model_load:
            return
        self._attempted_model_load = True

        if os.getenv("RAG_DISABLE_SENTENCE_TRANSFORMERS", "0") == "1":
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self._backend_name = f"sentence-transformers:{self.model_name}"
        except Exception:
            self._model = None
            self._backend_name = "hashing-local"


class KnowledgeBase:
    def __init__(self, documents_dir: Path, index_path: Path) -> None:
        self.documents_dir = documents_dir
        self.index_path = index_path
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = EmbeddingEngine()
        self.chunks: list[Chunk] = []
        self.index_backend = ""
        self.load()

    def load(self) -> None:
        if not self.index_path.exists():
            self.chunks = []
            self.index_backend = self.embedder.backend_name
            return

        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.chunks = []
            self.index_backend = self.embedder.backend_name
            return

        self.chunks = []
        for item in payload.get("chunks", []):
            self.chunks.append(
                Chunk(
                    id=item.get("id") or uuid.uuid4().hex,
                    document_name=item["document_name"],
                    chunk_number=int(item["chunk_number"]),
                    text=item["text"],
                    tokens=item.get("tokens") or tokenize(item["text"]),
                    page_number=item.get("page_number"),
                    embedding=item.get("embedding"),
                )
            )

        self.index_backend = payload.get("embedding_backend", "legacy")
        current_backend = self.embedder.backend_name
        missing_vectors = any(not chunk.embedding for chunk in self.chunks)
        if self.chunks and (missing_vectors or self.index_backend != current_backend):
            self._embed_all_chunks()
            self.save()
        else:
            self.index_backend = current_backend

    def save(self) -> None:
        payload = {
            "schema_version": 2,
            "embedding_backend": self.index_backend or self.embedder.backend_name,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_document(self, filename: str, content: bytes) -> int:
        safe_name = sanitize_filename(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type. Supported types: {supported}")

        path = self.documents_dir / safe_name
        path.write_bytes(content)
        try:
            sections = extract_sections(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise

        if not any(text.strip() for _, text in sections):
            path.unlink(missing_ok=True)
            raise ValueError("The uploaded document did not contain readable text.")

        self.chunks = [chunk for chunk in self.chunks if chunk.document_name != safe_name]
        new_chunks = make_chunks(sections, safe_name)
        embeddings = self.embedder.encode([chunk.text for chunk in new_chunks])
        for chunk, vector in zip(new_chunks, embeddings):
            chunk.embedding = vector

        self.chunks.extend(new_chunks)
        self.index_backend = self.embedder.backend_name
        self.save()
        return len(new_chunks)

    def remove_document(self, filename: str) -> bool:
        safe_name = sanitize_filename(filename)
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_name != safe_name]

        path = self.documents_dir / safe_name
        path.unlink(missing_ok=True)

        removed = before != len(self.chunks)
        if removed:
            self.save()
        return removed

    def rebuild(self) -> dict:
        self.chunks = []
        for path in sorted(self.documents_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    sections = extract_sections(path)
                except ValueError:
                    continue
                if any(text.strip() for _, text in sections):
                    self.chunks.extend(make_chunks(sections, path.name))

        self._embed_all_chunks()
        self.save()
        return self.stats()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []

        query_vector = self.embedder.encode([query])[0]
        results: list[SearchResult] = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = cosine_similarity(query_vector, chunk.embedding)
            if score > 0:
                results.append(SearchResult(chunk=chunk, score=round(score, 4)))

        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def answer(self, query: str, limit: int = 5) -> dict:
        results = self.search(query, limit=limit)
        if not results:
            return {
                "answer": "I could not find relevant information in the uploaded university documents.",
                "mode": "retrieval_augmented",
                "generation_backend": generation_backend_name(),
                "embedding_backend": self.index_backend,
                "source_count": 0,
                "sources": [],
            }

        answer, generator = generate_answer(query, results)
        sources = [format_source(result, index + 1) for index, result in enumerate(results)]

        return {
            "answer": answer,
            "mode": "retrieval_augmented",
            "generation_backend": generator,
            "embedding_backend": self.index_backend,
            "source_count": len(sources),
            "sources": sources,
        }

    def stats(self) -> dict:
        documents = sorted({chunk.document_name for chunk in self.chunks})
        total_tokens = sum(len(chunk.tokens) for chunk in self.chunks)
        return {
            "document_count": len(documents),
            "chunk_count": len(self.chunks),
            "token_count": total_tokens,
            "documents": documents,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "embedding_backend": self.index_backend or self.embedder.backend_name,
            "generation_backend": generation_backend_name(),
        }

    def _embed_all_chunks(self) -> None:
        embeddings = self.embedder.encode([chunk.text for chunk in self.chunks])
        for chunk, vector in zip(self.chunks, embeddings):
            chunk.embedding = vector
        self.index_backend = self.embedder.backend_name


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name) or f"document-{uuid.uuid4().hex}.txt"


def extract_sections(path: Path) -> list[tuple[int | None, str]]:
    extension = path.suffix.lower()
    raw = path.read_bytes()

    if extension == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise ValueError("PDF support requires pypdf. Run: pip install -r requirements.txt") from exc

        try:
            reader = PdfReader(path)
            return [
                (page_number, " ".join((page.extract_text() or "").split()))
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError("The PDF could not be read. Scanned PDFs may require OCR before upload.") from exc

    text = raw.decode("utf-8", errors="ignore")
    if extension == ".json":
        try:
            payload = json.loads(text)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return [(None, text)]


def make_chunks(
    sections: list[tuple[int | None, str]],
    document_name: str,
    chunk_size: int = 120,
    overlap: int = 25,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_number = 1

    for page_number, text in sections:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            tokens = tokenize(chunk_text)
            if tokens:
                chunks.append(
                    Chunk(
                        id=uuid.uuid4().hex,
                        document_name=document_name,
                        chunk_number=chunk_number,
                        page_number=page_number,
                        text=chunk_text,
                        tokens=tokens,
                    )
                )
                chunk_number += 1
            if end == len(words):
                break
            start = max(end - overlap, start + 1)

    return chunks


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def split_sentences(text: str) -> Iterable[str]:
    for sentence in SENTENCE_RE.split(text):
        cleaned = " ".join(sentence.split())
        if cleaned:
            yield cleaned


def hashing_embedding(text: str, dimensions: int = FALLBACK_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector

    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    counts = Counter(features)
    for feature, count in counts.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = -1.0 if (value >> 8) & 1 else 1.0
        vector[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    return sum(a * b for a, b in zip(first, second))


def generation_backend_name() -> str:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    return f"ollama:{model}" if model else "extractive-fallback"


def generate_answer(query: str, results: list[SearchResult]) -> tuple[str, str]:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if model:
        try:
            return generate_with_ollama(query, results, model), f"ollama:{model}"
        except (OSError, ValueError, urllib.error.URLError):
            pass

    return extractive_answer(query, results), "extractive-fallback"


def generate_with_ollama(query: str, results: list[SearchResult], model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    context_blocks = []
    for index, result in enumerate(results, start=1):
        page = f", page {result.chunk.page_number}" if result.chunk.page_number else ""
        context_blocks.append(
            f"[S{index}] {result.chunk.document_name}{page}, chunk {result.chunk.chunk_number}\n"
            f"{result.chunk.text}"
        )

    prompt = (
        "You are an Intelligent University Knowledge Assistant.\n"
        "Answer ONLY from the retrieved context below. If the context does not contain the answer, "
        "say that the information is not available in the uploaded documents. "
        "Use concise language and cite supporting sources with labels like [S1] or [S2].\n\n"
        "RETRIEVED CONTEXT:\n"
        + "\n\n".join(context_blocks)
        + f"\n\nUSER QUESTION:\n{query}\n\nANSWER:"
    )

    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    answer = (payload.get("response") or "").strip()
    if not answer:
        raise ValueError("Ollama returned an empty response.")
    return answer


def extractive_answer(query: str, results: list[SearchResult]) -> str:
    query_tokens = set(tokenize(query))
    selected: list[str] = []

    for result in results:
        for sentence in split_sentences(result.chunk.text):
            sentence_tokens = set(tokenize(sentence))
            if sentence_tokens & query_tokens:
                selected.append(sentence.strip())
            if len(selected) >= 4:
                break
        if len(selected) >= 4:
            break

    if not selected:
        selected = [results[0].chunk.text.strip()]
    return " ".join(dict.fromkeys(selected))


def confidence_label(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def format_source(result: SearchResult, rank: int) -> dict:
    return {
        "reference": f"S{rank}",
        "document": result.chunk.document_name,
        "page": result.chunk.page_number,
        "chunk": result.chunk.chunk_number,
        "score": result.score,
        "confidence": confidence_label(result.score),
        "excerpt": result.chunk.text[:420].strip(),
    }
