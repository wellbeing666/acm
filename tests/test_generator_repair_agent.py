import unittest
from pathlib import Path

from agent.generator_repair_agent import GeneratorRepairAgent, build_repair_prompt, read_generator_code
from agent.runtime import AgentTrace


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

    def test_agent_initializes_trace(self) -> None:
        spec = {
            "test_data_spec": {
                "version": "1.0",
                "problem_type": "math",
                "input_format": "n",
                "constraints": ["1 <= n <= 1"],
                "generation_notes": ["tiny"],
                "cases": [
                    {
                        "id": "TC001",
                        "name": "tiny",
                        "type": "boundary",
                        "purpose": "tiny",
                        "scale": "small",
                        "count": 1,
                        "parameters": {"n_range": "1"},
                        "construction": "Write n=1.",
                        "validity_checks": ["n == 1"],
                    }
                ],
            }
        }

        agent = GeneratorRepairAgent({"title": "A"}, spec, max_attempts=0)

        self.assertEqual(agent.trace.agent_name, "GeneratorRepairAgent")

    def test_agent_trace_records_events(self) -> None:
        trace = AgentTrace(agent_name="TestAgent")
        trace.add("observe", "ok", "saw something", path=Path("x"))

        data = trace.to_dict()

        self.assertEqual(data["agent_name"], "TestAgent")
        self.assertEqual(data["events"][0]["phase"], "observe")
        self.assertEqual(data["events"][0]["data"]["path"], "x")


if __name__ == "__main__":
    unittest.main()
