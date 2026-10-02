from pathlib import Path
import tempfile
import unittest

from rag_engine import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def test_indexes_and_answers_from_uploaded_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            chunks = kb.add_document(
                "handbook.txt",
                b"Attendance below 75 percent requires academic review. The library opens at 8 AM.",
            )

            self.assertEqual(chunks, 1)
            response = kb.answer("What happens when attendance is below 75 percent?")
            self.assertIn("Attendance below 75 percent", response["answer"])
            self.assertEqual(response["sources"][0]["document"], "handbook.txt")

    def test_replaces_existing_document_when_reuploaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb = KnowledgeBase(root / "documents", root / "index.json")

            kb.add_document("policy.txt", b"Old exam policy.")
            kb.add_document("policy.txt", b"New exam policy includes two internal tests.")

            stats = kb.stats()
            self.assertEqual(stats["document_count"], 1)
            self.assertIn("two internal tests", kb.answer("internal tests")["answer"])

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


if __name__ == "__main__":
    unittest.main()
