from pathlib import Path
import os
import tempfile
import unittest

os.environ["RAG_DISABLE_SENTENCE_TRANSFORMERS"] = "1"
os.environ.pop("OLLAMA_MODEL", None)

from rag_engine import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def test_indexes_vectors_and_answers_from_uploaded_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            chunks = kb.add_document(
                "handbook.txt",
                b"Attendance below 75 percent requires academic review. The library opens at 8 AM.",
            )

            self.assertEqual(chunks, 1)
            self.assertTrue(kb.chunks[0].embedding)
            response = kb.answer("What happens when attendance is below 75 percent?")
            self.assertIn("Attendance below 75 percent", response["answer"])
            self.assertEqual(response["sources"][0]["document"], "handbook.txt")
            self.assertEqual(response["sources"][0]["reference"], "S1")

    def test_replaces_existing_document_when_reuploaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            kb.add_document("policy.txt", b"Old exam policy.")
            kb.add_document("policy.txt", b"New exam policy includes two internal tests.")

            stats = kb.stats()
            self.assertEqual(stats["document_count"], 1)
            self.assertIn("two internal tests", kb.answer("internal tests")["answer"])

    def test_retrieves_from_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            kb.add_document("attendance.txt", b"Students need 75 percent attendance for exam eligibility.")
            kb.add_document("fees.txt", b"Students must pay examination fees before the hall ticket is issued.")

            response = kb.answer("What are the attendance and examination fee requirements?", limit=5)
            source_names = {source["document"] for source in response["sources"]}
            self.assertIn("attendance.txt", source_names)
            self.assertIn("fees.txt", source_names)

    def test_removes_document_and_rebuilds_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            kb.add_document("library.txt", b"The central library opens at 8 AM.")
            self.assertTrue(kb.remove_document("library.txt"))
            self.assertEqual(kb.stats()["document_count"], 0)

            (root / "documents" / "library.txt").write_text("The library closes at 8 PM.", encoding="utf-8")
            stats = kb.rebuild()
            self.assertEqual(stats["document_count"], 1)
            self.assertIn("8 PM", kb.answer("When does the library close?")["answer"])

    def test_persists_and_reloads_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            index = root / "index.json"
            kb = KnowledgeBase(documents, index)
            kb.add_document("admissions.txt", b"Applicants submit academic transcripts and identity proof.")

            reloaded = KnowledgeBase(documents, index)
            self.assertEqual(reloaded.stats()["document_count"], 1)
            self.assertTrue(reloaded.chunks[0].embedding)
            self.assertIn("transcripts", reloaded.answer("Which academic documents are required?")["answer"])


if __name__ == "__main__":
    unittest.main()
