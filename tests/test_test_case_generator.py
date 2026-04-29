import unittest

from src.schema import SchemaValidationError, validate_test_cases
from src.test_case_generator import build_test_case_prompt


class TestCaseGeneratorTests(unittest.TestCase):
    def test_validate_test_cases(self) -> None:
        data = {
            "test_cases": [
                {
                    "id": "TC001",
                    "name": "stress case",
                    "type": "stress",
                    "purpose": "Exercise the maximum input size.",
                    "input": "100000 0\n",
                    "reliability": "low",
                }
            ]
        }

        self.assertEqual(validate_test_cases(data), data)

    def test_validate_test_cases_rejects_missing_reliability(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_test_cases(
                {
                    "test_cases": [
                        {
                            "id": "TC001",
                            "name": "bad",
                            "type": "boundary",
                            "purpose": "bad",
                            "input": "1\n",
                        }
                    ]
                }
            )

    def test_build_test_case_prompt(self) -> None:
        prompt = build_test_case_prompt(
            {"title": "A", "input_format": "n"},
            {"test_plan": [{"name": "minimum", "type": "boundary", "purpose": "small"}]},
        )

        self.assertIn("problem_info", prompt)
        self.assertIn("test_plan", prompt)
        self.assertIn('"input_format": "n"', prompt)


if __name__ == "__main__":
    unittest.main()
