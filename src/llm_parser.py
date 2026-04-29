from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

try:
    from openai import OpenAI, OpenAIError
except ImportError:  # pragma: no cover - exercised only before dependencies are installed.
    OpenAI = None  # type: ignore[assignment]

    class OpenAIError(Exception):
        pass

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT_SECONDS,
    MAX_OUTPUT_TOKENS,
)
from src.schema import SchemaValidationError, validate_problem_analysis


SYSTEM_PROMPT = """
You are an expert ACM/ICPC competitive programming assistant.
Analyze the provided problem statement and extract a structured JSON summary for test case generation.

Rules:
- Return JSON only.
- Do not include markdown fences, comments, or explanatory text.
- Keep values concise but useful.
- If a detail is not explicitly given, infer cautiously from the statement and say "unknown" only when needed.
- Every field must be present.
- Array fields must always be JSON arrays of strings, even if there is only one item.

The JSON object must have exactly these fields:
{
  "title": "Clock Constellation",
  "problem_type": "graph shortest path",
  "algorithm_tags": ["dijkstra", "state compression", "shortest path"],
  "input_format": "First line contains n, k, m. The next lines describe constellations, visibility thresholds, and weighted undirected edges.",
  "output_format": "Print the minimum time to find constellation k, or -1 if impossible.",
  "constraints": ["1 <= n <= 100000", "1 <= k <= 7"],
  "corner_cases": ["target constellation is in the starting cluster", "some clusters are initially unobservable"]
}
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous response was empty or invalid. Output one complete valid JSON object only.
The first character must be { and the last character must be }.
""".strip()


class ProblemAnalysisError(Exception):
    """Base exception for ACM problem analysis failures."""


class MissingAPIKeyError(ProblemAnalysisError):
    """Raised when OPENAI_API_KEY is not configured."""


class MissingDependencyError(ProblemAnalysisError):
    """Raised when required Python packages are not installed."""


class ModelRequestError(ProblemAnalysisError):
    """Raised when the model request fails."""


class ModelJSONParseError(ProblemAnalysisError):
    """Raised when the model output cannot be parsed as valid JSON."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ModelSchemaError(ProblemAnalysisError):
    """Raised when the model output JSON does not match the required schema."""


def analyze_problem_statement(problem_statement: str) -> dict[str, Any]:
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
                    {"role": "user", "content": problem_statement},
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
            "Model returned an empty response after retries. "
            f"model={DEEPSEEK_MODEL}, finish_reason={last_finish_reason}, response_id={last_response_id}. "
            "DeepSeek JSON Output may occasionally return empty content; try running again or increasing MAX_OUTPUT_TOKENS.",
            raw_output,
        )

    parsed = parse_json_output(raw_output)

    try:
        return validate_problem_analysis(parsed)
    except SchemaValidationError as exc:
        raise ModelSchemaError(str(exc)) from exc


def parse_json_output(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except JSONDecodeError:
        pass

    extracted = extract_first_json_object(raw_output)
    if extracted is None:
        raise ModelJSONParseError("Model output is not valid JSON.", raw_output)

    try:
        return json.loads(extracted)
    except JSONDecodeError as exc:
        message = f"Extracted JSON object is invalid: {exc}"
        raise ModelJSONParseError(message, raw_output) from exc


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
