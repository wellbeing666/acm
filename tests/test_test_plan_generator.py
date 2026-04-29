import unittest

from src.schema import SchemaValidationError, validate_test_plan
from src.test_plan_generator import build_test_plan_prompt, format_retrieved_context


class TestPlanGeneratorTests(unittest.TestCase):
    def test_validate_test_plan(self) -> None:
        data = {
            "test_plan": [
                {
                    "name": "minimum case",
                    "type": "boundary",
                    "purpose": "Check the smallest valid input.",
                }
            ]
        }

        self.assertEqual(validate_test_plan(data), data)

    def test_validate_test_plan_rejects_missing_field(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_test_plan({"test_plan": [{"name": "x", "type": "boundary"}]})

    def test_format_retrieved_context(self) -> None:
        context = [
            {
                "metadata": {"source": "01_graph.md", "section": "边界情况"},
                "document": "- m=0 and target unreachable",
            }
        ]

        formatted = format_retrieved_context(context)

        self.assertIn("01_graph.md", formatted)
        self.assertIn("m=0", formatted)

    def test_build_test_plan_prompt(self) -> None:
        prompt = build_test_plan_prompt({"title": "A"}, "context")

        self.assertIn("problem_info", prompt)
        self.assertIn("retrieved_context", prompt)
        self.assertIn('"title": "A"', prompt)


if __name__ == "__main__":
    unittest.main()
