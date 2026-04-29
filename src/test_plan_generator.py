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
from src.schema import SchemaValidationError, validate_test_plan


SYSTEM_PROMPT = """
You are an expert ACM/ICPC test designer.
Create a concise test plan from structured problem information and retrieved RAG knowledge.

Rules:
- Return JSON only.
- Do not include markdown fences, comments, or explanatory text.
- The top-level JSON object must contain one key: "test_plan".
- "test_plan" must be an array of objects.
- Every object must contain exactly these string fields: "name", "type", "purpose".
- Use the retrieved context to include common strategies, boundary cases, and counterexamples.
- Prefer 6 to 10 focused plan items.
- Keep each purpose specific enough for later test case generation.

Example:
{
  "test_plan": [
    {
      "name": "minimum reachable case",
      "type": "boundary",
      "purpose": "Verify the answer when the smallest valid input already satisfies the target condition."
    }
  ]
}
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous response was empty or invalid. Output one complete valid JSON object only.
The first character must be { and the last character must be }.
""".strip()


def generate_test_plan(problem_info: dict[str, Any], retrieved_context: list[dict[str, Any]] | str) -> dict[str, Any]:
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

    user_prompt = build_test_plan_prompt(problem_info, retrieved_context)
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
            "Model returned an empty test plan response after retries. "
            f"model={DEEPSEEK_MODEL}, finish_reason={last_finish_reason}, response_id={last_response_id}.",
            raw_output,
        )

    parsed = parse_json_output(raw_output)

    try:
        return validate_test_plan(parsed)
    except SchemaValidationError as exc:
        raise ModelSchemaError(str(exc)) from exc


def build_test_plan_prompt(problem_info: dict[str, Any], retrieved_context: list[dict[str, Any]] | str) -> str:
    return (
        "problem_info:\n"
        f"{json.dumps(problem_info, ensure_ascii=False, indent=2)}\n\n"
        "retrieved_context:\n"
        f"{format_retrieved_context(retrieved_context)}"
    )


def format_retrieved_context(retrieved_context: list[dict[str, Any]] | str) -> str:
    if isinstance(retrieved_context, str):
        return retrieved_context

    blocks: list[str] = []
    for index, item in enumerate(retrieved_context, start=1):
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "unknown")
        document = item.get("document", "")
        blocks.append(f"[{index}] source={source}, section={section}\n{document}")

    return "\n\n".join(blocks)
