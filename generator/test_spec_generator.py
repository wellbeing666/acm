from __future__ import annotations

import json
from typing import Any

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT_SECONDS,
    TEST_SPEC_MAX_OUTPUT_TOKENS,
)
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
    OpenAI,
    OpenAIError,
    parse_json_output,
)
from src.schema import SchemaValidationError, validate_test_data_spec
from src.test_plan_generator import format_retrieved_context


SYSTEM_PROMPT = """
You are an ACM/ICPC test data specification designer.
Create a TestDataSpec from problem_info, test_plan, and retrieved_context.

Rules:
- Return JSON only.
- Do not include markdown fences, comments, or explanatory text.
- Do not generate concrete ACM input data.
- Do not generate expected output.
- The top-level JSON object must contain exactly one key: "test_data_spec".
- TestDataSpec describes how a local generator should create tests, not the tests themselves.
- Use retrieved_context to include boundary, stress, random, and counterexample-oriented generation strategies.
- Keep cases compact and actionable for code generation.
- Prefer 6 to 8 cases total, merging similar test_plan items when needed.
- Keep every string under 160 characters.
- parameters must describe ranges, shapes, and recipes. Do not include concrete arrays such as x_i, y_i, or edges.
- Forbidden parameter keys: x_i, y_i, edges, edge_list, input, output, expected_output, test_data.
- Use abstract keys instead: x_pattern, y_pattern, edge_pattern, node_range, edge_count, weight_pattern.
- Do not mention correct answers, output values, or expected output in any field.
- For concrete small cases, describe them as recipes, not raw input.

Required JSON shape:
{
  "test_data_spec": {
    "version": "1.0",
    "problem_type": "string",
    "input_format": "string",
    "constraints": ["string"],
    "generation_notes": ["string"],
    "cases": [
      {
        "id": "TC001",
        "name": "string",
        "type": "boundary|random|stress|counterexample|sample_like",
        "purpose": "string",
        "scale": "small|medium|large|max",
        "count": 1,
        "parameters": {
          "n_range": "1",
          "shape": "chain graph",
          "edge_weight_pattern": "all maximum"
        },
        "construction": "Describe the algorithm for generating this category without listing raw input.",
        "validity_checks": ["Every generated input must satisfy ..."]
      }
    ]
  }
}
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous response was empty or invalid. Output one complete valid JSON object only.
The first character must be { and the last character must be }.
Do not include concrete test input or expected output.
Make the JSON shorter than before: at most 8 cases, concise strings, no raw arrays.
Never use these parameter keys: x_i, y_i, edges, edge_list, input, output, expected_output, test_data.
""".strip()


def generate_test_data_spec(
    problem_info: dict[str, Any],
    test_plan: dict[str, Any],
    retrieved_context: list[dict[str, Any]] | str,
) -> dict[str, Any]:
    if OpenAI is None:
        raise MissingDependencyError("The 'openai' package is not installed. Run: pip install -r requirements.txt")

    if not DEEPSEEK_API_KEY:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set. Set it in your environment before running this command."
        )

    user_prompt = build_test_data_spec_prompt(problem_info, test_plan, retrieved_context)
    last_raw_output = ""
    last_error = ""

    for attempt in range(DEEPSEEK_MAX_RETRIES + 1):
        raw_output = request_json_from_model(user_prompt, retry_note=last_error if attempt > 0 else "")
        last_raw_output = raw_output
        try:
            parsed = parse_json_output(raw_output)
            parsed = normalize_test_data_spec(parsed)
            return validate_test_data_spec(parsed)
        except (ModelJSONParseError, SchemaValidationError) as exc:
            last_error = str(exc)

    raise ModelJSONParseError(
        f"Model returned invalid TestDataSpec JSON after retries. Last error: {last_error}",
        last_raw_output,
    )


def build_test_data_spec_prompt(
    problem_info: dict[str, Any],
    test_plan: dict[str, Any],
    retrieved_context: list[dict[str, Any]] | str,
) -> str:
    return (
        "problem_info:\n"
        f"{json.dumps(problem_info, ensure_ascii=False, indent=2)}\n\n"
        "test_plan:\n"
        f"{json.dumps(test_plan, ensure_ascii=False, indent=2)}\n\n"
        "retrieved_context:\n"
        f"{format_retrieved_context(retrieved_context)}"
    )


def normalize_test_data_spec(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    spec = data.get("test_data_spec") if isinstance(data.get("test_data_spec"), dict) else data
    cases = spec.get("cases") if isinstance(spec, dict) else None
    if not isinstance(cases, list):
        return data

    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("parameters"), dict):
            continue
        item["parameters"] = normalize_spec_parameters(item["parameters"])

    return data


def normalize_spec_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    replacements = {
        "x_i": "x_pattern",
        "y_i": "y_pattern",
        "edges": "edge_pattern",
        "edge_list": "edge_pattern",
        "input": "input_recipe",
        "output": "output_omitted",
        "expected_output": "output_omitted",
        "test_data": "generation_recipe",
    }
    normalized: dict[str, Any] = {}

    for key, value in parameters.items():
        normalized_key = replacements.get(key, key)
        normalized[normalized_key] = summarize_parameter_value(value)

    return normalized


def summarize_parameter_value(value: Any) -> Any:
    if isinstance(value, list):
        return f"omitted raw list with {len(value)} item(s); generate algorithmically from construction"
    if isinstance(value, dict):
        return {str(key): summarize_parameter_value(child) for key, child in value.items()}
    return value


def request_json_from_model(user_prompt: str, retry_note: str = "") -> str:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
    )

    raw_output = ""
    last_finish_reason = "unknown"
    last_response_id = "unknown"

    system_prompt = SYSTEM_PROMPT
    if retry_note:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{RETRY_PROMPT_SUFFIX}\nLast validation error: {retry_note}"

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=TEST_SPEC_MAX_OUTPUT_TOKENS,
            extra_body={"thinking": {"type": DEEPSEEK_THINKING}},
        )

        choice = response.choices[0]
        last_finish_reason = choice.finish_reason or "unknown"
        last_response_id = getattr(response, "id", "unknown")
        raw_output = (choice.message.content or "").strip()

        if raw_output:
            return raw_output
    except OpenAIError as exc:
        raise ModelRequestError(f"DeepSeek request failed: {exc}") from exc

    raise ModelJSONParseError(
        "Model returned an empty TestDataSpec response after retries. "
        f"model={DEEPSEEK_MODEL}, finish_reason={last_finish_reason}, response_id={last_response_id}.",
        raw_output,
    )
