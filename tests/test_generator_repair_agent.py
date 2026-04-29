import unittest
from pathlib import Path

from agent.generator_repair_agent import build_repair_prompt, read_generator_code


class GeneratorRepairAgentTests(unittest.TestCase):
    def test_build_repair_prompt_contains_error_and_code(self) -> None:
        prompt = build_repair_prompt(
            problem_info={"title": "A"},
            test_data_spec={"test_data_spec": {"cases": []}},
            current_code="print('bad')",
            error_message="SyntaxError",
            attempt=1,
        )

        self.assertIn("generator_error", prompt)
        self.assertIn("SyntaxError", prompt)
        self.assertIn("current_generator_code", prompt)
        self.assertIn("print('bad')", prompt)

    def test_read_generator_code_missing_file_returns_empty_string(self) -> None:
        self.assertEqual(read_generator_code(Path("missing_generator_for_test.py")), "")


if __name__ == "__main__":
    unittest.main()
