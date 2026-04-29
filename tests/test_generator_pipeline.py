import unittest

from generator.generator_code_builder import validate_generator_source
from generator.test_spec_generator import normalize_test_data_spec
from src.schema import SchemaValidationError, validate_generator_code, validate_test_data_spec


class GeneratorPipelineTests(unittest.TestCase):
    def test_validate_test_data_spec(self) -> None:
        spec = {
            "test_data_spec": {
                "version": "1.0",
                "problem_type": "graph",
                "input_format": "n m followed by edges",
                "constraints": ["1 <= n <= 100000"],
                "generation_notes": ["Generate graphs algorithmically."],
                "cases": [
                    {
                        "id": "TC001",
                        "name": "single node",
                        "type": "boundary",
                        "purpose": "Check minimal graph.",
                        "scale": "small",
                        "count": 1,
                        "parameters": {"n": 1, "m": 0},
                        "construction": "Generate the smallest graph.",
                        "validity_checks": ["n == 1", "m == 0"],
                    }
                ],
            }
        }

        self.assertEqual(validate_test_data_spec(spec), spec)

    def test_validate_test_data_spec_rejects_direct_input(self) -> None:
        spec = {
            "test_data_spec": {
                "version": "1.0",
                "problem_type": "math",
                "input_format": "n",
                "constraints": ["1 <= n <= 10"],
                "generation_notes": ["No direct data."],
                "cases": [
                    {
                        "id": "TC001",
                        "name": "bad",
                        "type": "boundary",
                        "purpose": "bad",
                        "scale": "small",
                        "count": 1,
                        "parameters": {},
                        "construction": "bad",
                        "validity_checks": [],
                        "input": "1\n",
                    }
                ],
            }
        }

        with self.assertRaises(SchemaValidationError):
            validate_test_data_spec(spec)

    def test_validate_test_data_spec_rejects_direct_parameter_arrays(self) -> None:
        spec = {
            "test_data_spec": {
                "version": "1.0",
                "problem_type": "graph",
                "input_format": "n m edges",
                "constraints": ["1 <= n <= 10"],
                "generation_notes": ["No direct arrays."],
                "cases": [
                    {
                        "id": "TC001",
                        "name": "bad arrays",
                        "type": "boundary",
                        "purpose": "bad",
                        "scale": "small",
                        "count": 1,
                        "parameters": {"edges": [[1, 2, 3]]},
                        "construction": "bad",
                        "validity_checks": ["valid"],
                    }
                ],
            }
        }

        with self.assertRaises(SchemaValidationError):
            validate_test_data_spec(spec)

    def test_normalize_test_data_spec_rewrites_direct_parameter_keys(self) -> None:
        spec = {
            "test_data_spec": {
                "version": "1.0",
                "problem_type": "graph",
                "input_format": "n m edges",
                "constraints": ["1 <= n <= 10"],
                "generation_notes": ["Normalize raw fields."],
                "cases": [
                    {
                        "id": "TC001",
                        "name": "raw edge field",
                        "type": "boundary",
                        "purpose": "normalization",
                        "scale": "small",
                        "count": 1,
                        "parameters": {"edges": [[1, 2, 3]], "x_i": [1, 0]},
                        "construction": "Generate these structures algorithmically.",
                        "validity_checks": ["valid"],
                    }
                ],
            }
        }

        normalized = normalize_test_data_spec(spec)
        parameters = normalized["test_data_spec"]["cases"][0]["parameters"]

        self.assertIn("edge_pattern", parameters)
        self.assertIn("x_pattern", parameters)
        self.assertNotIn("edges", parameters)
        validate_test_data_spec(normalized)

    def test_validate_generator_code_rejects_expected_output(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_generator_code({"generator_code": "expected_output = None"})

    def test_validate_generator_source_rejects_forbidden_import(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_generator_source("import subprocess\n")

    def test_validate_generator_source_rejects_out_file_literal(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_generator_source("path = 'answer.out'\n")

    def test_validate_generator_source_allows_out_in_comment(self) -> None:
        validate_generator_source("# Do not create .out files\n")

    def test_validate_generator_source_accepts_basic_code(self) -> None:
        validate_generator_source(
            "from pathlib import Path\n"
            "import argparse\n"
            "def main():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )


if __name__ == "__main__":
    unittest.main()
