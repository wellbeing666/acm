import unittest

from src.llm_parser import ModelJSONParseError, parse_json_output
from src.schema import SchemaValidationError, validate_problem_analysis


class ParserTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        self.assertEqual(parse_json_output('{"title": "A"}'), {"title": "A"})

    def test_parse_json_wrapped_in_text(self) -> None:
        output = 'Here is the result:\n{"title": "A", "nested": {"x": 1}}\nDone.'
        self.assertEqual(parse_json_output(output), {"title": "A", "nested": {"x": 1}})

    def test_parse_invalid_json_raises(self) -> None:
        with self.assertRaises(ModelJSONParseError):
            parse_json_output("not json")


class SchemaTests(unittest.TestCase):
    def test_validate_expected_schema(self) -> None:
        data = {
            "title": "A",
            "problem_type": "math",
            "algorithm_tags": ["bruteforce"],
            "input_format": "n",
            "output_format": "n",
            "constraints": ["1 <= n <= 10"],
            "corner_cases": ["n = 1"],
        }
        self.assertEqual(validate_problem_analysis(data), data)

    def test_validate_missing_field_raises(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_problem_analysis({"title": "A"})


if __name__ == "__main__":
    unittest.main()
