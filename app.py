from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote

from rag_engine import KnowledgeBase


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB = KnowledgeBase(DATA_DIR / "documents", DATA_DIR / "index.json")


class AssistantHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(BASE_DIR / "templates" / "index.html")
        elif parsed.path == "/static/styles.css":
            self._send_file(BASE_DIR / "static" / "styles.css", "text/css")
        elif parsed.path == "/api/stats":
            self._send_json(KB.stats())
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok", "service": "knowledge-vector-db"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload()
        elif parsed.path == "/api/ask":
            self._handle_ask()
        elif parsed.path == "/api/rebuild":
            self._send_json({"message": "Repository rebuilt.", "stats": KB.rebuild()})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/documents/"):
            filename = unquote(parsed.path.removeprefix("/api/documents/"))
            if KB.remove_document(filename):
                self._send_json({"message": f"Removed {filename}.", "stats": KB.stats()})
            else:
                self._send_json({"error": "Document not found."}, HTTPStatus.NOT_FOUND)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_upload(self) -> None:
        upload = self._read_multipart_file("document")
        if upload is None:
            self._send_json({"error": "Choose a document to upload."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            chunk_count = KB.add_document(upload["filename"], upload["content"])
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"message": f"Indexed {chunk_count} chunks from {upload['filename']}.", "stats": KB.stats()})

    def _handle_ask(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON request body."}, HTTPStatus.BAD_REQUEST)
                return
        else:
            payload = {key: value[0] for key, value in parse_qs(body).items()}

        question = (payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Enter a question."}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json(KB.answer(question))

    def _send_html(self, path: Path) -> None:
        self._send_file(path, "text/html; charset=utf-8")

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_multipart_file(self, field_name: str) -> dict | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_length)
        headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        message = BytesParser(policy=default).parsebytes(headers + body)

        if not message.is_multipart():
            return None

        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            if disposition == "form-data" and name == field_name and filename:
                content = part.get_payload(decode=True) or b""
                return {"filename": filename, "content": content}

        return None


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AssistantHandler)
    print("Intelligent University Knowledge Assistant running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
