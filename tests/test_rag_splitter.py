import unittest
from pathlib import Path

from src.rag_splitter import load_markdown_chunks, parse_front_matter, split_text


class RAGSplitterTests(unittest.TestCase):
    def test_parse_front_matter(self) -> None:
        metadata, body = parse_front_matter(
            "---\n"
            "title: 图论题测试策略\n"
            "problem_type: graph\n"
            "algorithm_tags: [shortest_path, dijkstra]\n"
            "---\n"
            "# Body\n"
        )

        self.assertEqual(metadata["title"], "图论题测试策略")
        self.assertEqual(metadata["problem_type"], "graph")
        self.assertEqual(metadata["algorithm_tags"], ["shortest_path", "dijkstra"])
        self.assertEqual(body.strip(), "# Body")

    def test_split_text_keeps_overlap(self) -> None:
        chunks = split_text("a" * 10 + "\n\n" + "b" * 10 + "\n\n" + "c" * 10, chunk_size=24, chunk_overlap=3)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("bbb"))

    def test_load_markdown_chunks(self) -> None:
        root = Path(__file__).parent / "fixtures" / "sample_knowledge"
        chunks = load_markdown_chunks(root, chunk_size=200, chunk_overlap=20)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["source"], "sample.md")
        self.assertEqual(chunks[1].metadata["section"], "常见测试策略")


if __name__ == "__main__":
    unittest.main()
