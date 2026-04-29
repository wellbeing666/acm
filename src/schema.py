from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = (
    "title",
    "problem_type",
    "algorithm_tags",
    "input_format",
    "output_format",
    "constraints",
    "corner_cases",
)

TEST_PLAN_ITEM_FIELDS = ("name", "type", "purpose")
TEST_CASE_ITEM_FIELDS = ("id", "name", "type", "purpose", "input", "reliability")
TEST_DATA_SPEC_ROOT_FIELDS = (
    "version",
    "problem_type",
    "input_format",
    "constraints",
    "generation_notes",
    "cases",
)
TEST_DATA_SPEC_CASE_FIELDS = (
    "id",
    "name",
    "type",
    "purpose",
    "scale",
    "count",
    "parameters",
    "construction",
    "validity_checks",
)


PROBLEM_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Problem title. Use an inferred concise title if the statement has no explicit title.",
        },
        "problem_type": {
            "type": "string",
            "description": "High-level competitive programming problem type, such as graph, dp, greedy, math, string, simulation, data_structure, geometry, or mixed.",
        },
        "algorithm_tags": {
            "type": "array",
            "description": "Likely algorithm or data-structure tags.",
            "items": {"type": "string"},
        },
        "input_format": {
            "type": "string",
            "description": "A concise description of the expected input format.",
        },
        "output_format": {
            "type": "string",
            "description": "A concise description of the expected output format.",
        },
        "constraints": {
            "type": "array",
            "description": "All explicit and strongly implied constraints that matter for test generation.",
            "items": {"type": "string"},
        },
        "corner_cases": {
            "type": "array",
            "description": "Important edge cases for generating tests.",
            "items": {"type": "string"},
        },
    },
    "required": list(REQUIRED_FIELDS),
    "additionalProperties": False,
}


class SchemaValidationError(ValueError):
    """Raised when the model output is valid JSON but does not match the app schema."""


def validate_problem_analysis(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaValidationError("Model output must be a JSON object.")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        joined = ", ".join(missing_fields)
        raise SchemaValidationError(f"Model output is missing required fields: {joined}")

    string_fields = ("title", "problem_type", "input_format", "output_format")
    for field in string_fields:
        if not isinstance(data[field], str):
            raise SchemaValidationError(f"Field '{field}' must be a string.")

    list_fields = ("algorithm_tags", "constraints", "corner_cases")
    for field in list_fields:
        if not isinstance(data[field], list):
            raise SchemaValidationError(f"Field '{field}' must be a list.")
        if not all(isinstance(item, str) for item in data[field]):
            raise SchemaValidationError(f"Every item in field '{field}' must be a string.")

    return {field: data[field] for field in REQUIRED_FIELDS}


def validate_test_plan(data: Any) -> dict[str, list[dict[str, str]]]:
    if isinstance(data, list):
        plan_items = data
    elif isinstance(data, dict) and isinstance(data.get("test_plan"), list):
        plan_items = data["test_plan"]
    else:
        raise SchemaValidationError("Test plan output must be a JSON object with a 'test_plan' list.")

    if not plan_items:
        raise SchemaValidationError("Field 'test_plan' must contain at least one item.")

    validated_items: list[dict[str, str]] = []
    for index, item in enumerate(plan_items):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"Test plan item {index} must be a JSON object.")

        missing_fields = [field for field in TEST_PLAN_ITEM_FIELDS if field not in item]
        if missing_fields:
            joined = ", ".join(missing_fields)
            raise SchemaValidationError(f"Test plan item {index} is missing required fields: {joined}")

        validated_item: dict[str, str] = {}
        for field in TEST_PLAN_ITEM_FIELDS:
            value = item[field]
            if not isinstance(value, str):
                raise SchemaValidationError(f"Field '{field}' in test plan item {index} must be a string.")
            validated_item[field] = value

        validated_items.append(validated_item)

    return {"test_plan": validated_items}


def validate_test_cases(data: Any) -> dict[str, list[dict[str, str]]]:
    if isinstance(data, list):
        test_cases = data
    elif isinstance(data, dict) and isinstance(data.get("test_cases"), list):
        test_cases = data["test_cases"]
    else:
        raise SchemaValidationError("Test case output must be a JSON object with a 'test_cases' list.")

    if not test_cases:
        raise SchemaValidationError("Field 'test_cases' must contain at least one item.")

    validated_cases: list[dict[str, str]] = []
    for index, item in enumerate(test_cases):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"Test case item {index} must be a JSON object.")

        missing_fields = [field for field in TEST_CASE_ITEM_FIELDS if field not in item]
        if missing_fields:
            joined = ", ".join(missing_fields)
            raise SchemaValidationError(f"Test case item {index} is missing required fields: {joined}")

        validated_item: dict[str, str] = {}
        for field in TEST_CASE_ITEM_FIELDS:
            value = item[field]
            if not isinstance(value, str):
                raise SchemaValidationError(f"Field '{field}' in test case item {index} must be a string.")
            validated_item[field] = value

        validated_cases.append(validated_item)

    return {"test_cases": validated_cases}


def validate_test_data_spec(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("test_data_spec"), dict):
        spec = data["test_data_spec"]
    elif isinstance(data, dict):
        spec = data
    else:
        raise SchemaValidationError("TestDataSpec output must be a JSON object.")

    missing_fields = [field for field in TEST_DATA_SPEC_ROOT_FIELDS if field not in spec]
    if missing_fields:
        joined = ", ".join(missing_fields)
        raise SchemaValidationError(f"TestDataSpec is missing required fields: {joined}")

    for field in ("version", "problem_type", "input_format"):
        if not isinstance(spec[field], str):
            raise SchemaValidationError(f"Field '{field}' in TestDataSpec must be a string.")

    for field in ("constraints", "generation_notes"):
        if not isinstance(spec[field], list) or not all(isinstance(item, str) for item in spec[field]):
            raise SchemaValidationError(f"Field '{field}' in TestDataSpec must be a list of strings.")

    if not isinstance(spec["cases"], list) or not spec["cases"]:
        raise SchemaValidationError("Field 'cases' in TestDataSpec must be a non-empty list.")

    validated_cases: list[dict[str, Any]] = []
    for index, item in enumerate(spec["cases"]):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"TestDataSpec case {index} must be a JSON object.")

        forbidden_fields = [field for field in ("input", "output", "expected_output", "test_data") if field in item]
        if forbidden_fields:
            joined = ", ".join(forbidden_fields)
            raise SchemaValidationError(
                f"TestDataSpec case {index} must not contain direct test data fields: {joined}"
            )

        missing_case_fields = [field for field in TEST_DATA_SPEC_CASE_FIELDS if field not in item]
        if missing_case_fields:
            joined = ", ".join(missing_case_fields)
            raise SchemaValidationError(f"TestDataSpec case {index} is missing required fields: {joined}")

        validated_item: dict[str, Any] = {}
        for field in ("id", "name", "type", "purpose", "scale", "construction"):
            if not isinstance(item[field], str):
                raise SchemaValidationError(f"Field '{field}' in TestDataSpec case {index} must be a string.")
            validated_item[field] = item[field]

        if not isinstance(item["count"], int) or item["count"] < 1:
            raise SchemaValidationError(f"Field 'count' in TestDataSpec case {index} must be a positive integer.")
        validated_item["count"] = item["count"]

        if not isinstance(item["parameters"], dict):
            raise SchemaValidationError(f"Field 'parameters' in TestDataSpec case {index} must be an object.")
        validate_spec_parameters(item["parameters"], index)
        validated_item["parameters"] = item["parameters"]

        if not isinstance(item["validity_checks"], list) or not all(
            isinstance(check, str) for check in item["validity_checks"]
        ):
            raise SchemaValidationError(
                f"Field 'validity_checks' in TestDataSpec case {index} must be a list of strings."
            )
        validated_item["validity_checks"] = item["validity_checks"]

        validated_cases.append(validated_item)

    return {
        "test_data_spec": {
            "version": spec["version"],
            "problem_type": spec["problem_type"],
            "input_format": spec["input_format"],
            "constraints": spec["constraints"],
            "generation_notes": spec["generation_notes"],
            "cases": validated_cases,
        }
    }


def validate_spec_parameters(parameters: dict[str, Any], case_index: int) -> None:
    forbidden_direct_data_keys = {
        "x_i",
        "y_i",
        "edges",
        "edge_list",
        "input",
        "output",
        "expected_output",
        "test_data",
    }
    found = sorted(key for key in parameters if key in forbidden_direct_data_keys)
    if found:
        joined = ", ".join(found)
        raise SchemaValidationError(
            f"Field 'parameters' in TestDataSpec case {case_index} contains direct data keys: {joined}"
        )

    for key, value in parameters.items():
        if isinstance(value, list):
            raise SchemaValidationError(
                f"Field 'parameters.{key}' in TestDataSpec case {case_index} must describe data, not list it."
            )


def validate_generator_code(data: Any) -> dict[str, str]:
    if not isinstance(data, dict) or not isinstance(data.get("generator_code"), str):
        raise SchemaValidationError("Generator code output must be a JSON object with a 'generator_code' string.")

    code = data["generator_code"].strip()
    if not code:
        raise SchemaValidationError("Field 'generator_code' must not be empty.")
    if "```" in code:
        raise SchemaValidationError("Field 'generator_code' must contain raw Python code without markdown fences.")
    if "expected_output" in code:
        raise SchemaValidationError("Generator code must not produce or mention expected_output.")

    return {"generator_code": code}
