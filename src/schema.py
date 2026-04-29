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
