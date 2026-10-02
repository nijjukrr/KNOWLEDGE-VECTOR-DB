from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    document_name: str
    chunk_number: int
    text: str
    tokens: list[str]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, documents_dir: Path, index_path: Path) -> None:
        self.documents_dir = documents_dir
        self.index_path = index_path
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self.document_frequencies: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not self.index_path.exists():
            self.chunks = []
            self.document_frequencies = {}
            return

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        self.document_frequencies = payload.get("document_frequencies", {})

    def save(self) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "document_frequencies": self.document_frequencies,
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
        text = extract_text(path)
        if not text.strip():
            raise ValueError("The uploaded document did not contain readable text.")

        self.chunks = [chunk for chunk in self.chunks if chunk.document_name != safe_name]
        new_chunks = make_chunks(text, safe_name)
        self.chunks.extend(new_chunks)
        self._rebuild_document_frequencies()
        self.save()
        return len(new_chunks)

    def remove_document(self, filename: str) -> bool:
        safe_name = sanitize_filename(filename)
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_name != safe_name]

        path = self.documents_dir / safe_name
        if path.exists():
            path.unlink()

        removed = before != len(self.chunks)
        if removed:
            self._rebuild_document_frequencies()
            self.save()
        return removed

    def rebuild(self) -> dict:
        self.chunks = []
        for path in sorted(self.documents_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                text = extract_text(path)
                if text.strip():
                    self.chunks.extend(make_chunks(text, path.name))
        self._rebuild_document_frequencies()
        self.save()
        return self.stats()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        query_counts = Counter(query_tokens)
        scores: list[SearchResult] = []
        chunk_count = len(self.chunks)

        for chunk in self.chunks:
            token_counts = Counter(chunk.tokens)
            score = 0.0
            length_norm = math.sqrt(sum(count * count for count in token_counts.values())) or 1.0

            for token, query_count in query_counts.items():
                term_frequency = token_counts[token] / length_norm
                inverse_document_frequency = math.log((1 + chunk_count) / (1 + self.document_frequencies.get(token, 0))) + 1
                score += query_count * term_frequency * inverse_document_frequency

            if score > 0:
                scores.append(SearchResult(chunk=chunk, score=round(score, 4)))

        return sorted(scores, key=lambda result: result.score, reverse=True)[:limit]

    def answer(self, query: str, limit: int = 5) -> dict:
        results = self.search(query, limit=limit)
        if not results:
            return {
                "answer": "I could not find relevant information in the uploaded university documents.",
                "sources": [],
            }

        query_tokens = set(tokenize(query))
        selected_sentences: list[str] = []

        for result in results:
            for sentence in split_sentences(result.chunk.text):
                sentence_tokens = set(tokenize(sentence))
                if sentence_tokens & query_tokens:
                    selected_sentences.append(sentence.strip())
                if len(selected_sentences) >= 4:
                    break
            if len(selected_sentences) >= 4:
                break

        if not selected_sentences:
            selected_sentences = [results[0].chunk.text.strip()]

        sources = [
            {
                "document": result.chunk.document_name,
                "chunk": result.chunk.chunk_number,
                "score": result.score,
                "confidence": confidence_label(result.score),
                "excerpt": result.chunk.text[:360].strip(),
            }
            for result in results
        ]

        return {
            "answer": " ".join(selected_sentences),
            "mode": "retrieval_augmented",
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
        }

    def _rebuild_document_frequencies(self) -> None:
        frequencies: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            for token in set(chunk.tokens):
                frequencies[token] += 1
        self.document_frequencies = dict(frequencies)


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name) or f"document-{uuid.uuid4().hex}.txt"


def extract_text(path: Path) -> str:
    extension = path.suffix.lower()
    raw = path.read_bytes()

    if extension == ".pdf":
        text = raw.decode("latin-1", errors="ignore")
        text = re.sub(r"\\[rn]", " ", text)
        text = re.sub(r"[^ -~]+", " ", text)
        return " ".join(text.split())

    return raw.decode("utf-8", errors="ignore")


def make_chunks(text: str, document_name: str, chunk_size: int = 120, overlap: int = 25) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    chunk_number = 1

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        tokens = tokenize(chunk_text)
        if tokens:
            chunks.append(
                Chunk(
                    id=uuid.uuid4().hex,
                    document_name=document_name,
                    chunk_number=chunk_number,
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


def confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"
