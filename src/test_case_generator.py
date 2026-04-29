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
    MAX_OUTPUT_TOKENS,
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
from src.schema import SchemaValidationError, validate_test_cases


SYSTEM_PROMPT = """
You are an expert ACM/ICPC test case generator.
Create concrete ACM test cases from structured problem information and a test plan.

Rules:
- Return JSON only.
- Do not include markdown fences, comments, or explanatory text.
- The top-level JSON object must contain one key: "test_cases".
- "test_cases" must be an array of objects.
- Every object must contain exactly these fields: "id", "name", "type", "purpose", "input", "reliability".
- All fields must be strings.
- Do not generate expected output. Do not include an "expected_output" field.
- Generate valid input according to problem_info.input_format and problem_info.constraints.
- Prefer concrete test input data that can later be fed into a standard solution.
- Use escaped newlines inside JSON strings for multi-line ACM input and output.
- reliability should be one of: "high", "medium", "low".

Example:
{
  "test_cases": [
    {
      "id": "TC001",
      "name": "minimum case",
      "type": "boundary",
      "purpose": "Check the smallest valid input.",
      "input": "1\\n",
      "reliability": "high"
    }
  ]
}
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous response was empty or invalid. Output one complete valid JSON object only.
The first character must be { and the last character must be }.
Do not include expected_output.
""".strip()


def generate_test_cases(problem_info: dict[str, Any], test_plan: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if OpenAI is None:
        raise MissingDependencyError("The 'openai' package is not installed. Run: pip install -r requirements.txt")

    if not DEEPSEEK_API_KEY:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set. Set it in your environment before running this command."
        )

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
    )

    user_prompt = build_test_case_prompt(problem_info, test_plan)
    raw_output = ""
    last_finish_reason = "unknown"
    last_response_id = "unknown"

    try:
        for attempt in range(DEEPSEEK_MAX_RETRIES + 1):
            system_prompt = SYSTEM_PROMPT
            if attempt > 0:
                system_prompt = f"{SYSTEM_PROMPT}\n\n{RETRY_PROMPT_SUFFIX}"

            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"thinking": {"type": DEEPSEEK_THINKING}},
            )

            choice = response.choices[0]
            last_finish_reason = choice.finish_reason or "unknown"
            last_response_id = getattr(response, "id", "unknown")
            raw_output = (choice.message.content or "").strip()

            if raw_output:
                break
    except OpenAIError as exc:
        raise ModelRequestError(f"DeepSeek request failed: {exc}") from exc

    if not raw_output:
        raise ModelJSONParseError(
            "Model returned an empty test case response after retries. "
            f"model={DEEPSEEK_MODEL}, finish_reason={last_finish_reason}, response_id={last_response_id}.",
            raw_output,
        )

    parsed = parse_json_output(raw_output)

    try:
        return validate_test_cases(parsed)
    except SchemaValidationError as exc:
        raise ModelSchemaError(str(exc)) from exc


def build_test_case_prompt(problem_info: dict[str, Any], test_plan: dict[str, Any] | list[dict[str, Any]]) -> str:
    return (
        "problem_info:\n"
        f"{json.dumps(problem_info, ensure_ascii=False, indent=2)}\n\n"
        "test_plan:\n"
        f"{json.dumps(test_plan, ensure_ascii=False, indent=2)}"
    )
