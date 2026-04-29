from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT_SECONDS,
    GENERATOR_CODE_MAX_OUTPUT_TOKENS,
    GENERATOR_REPAIR_MAX_ATTEMPTS,
    GENERATED_GENERATOR_PATH,
    TEST_DATA_DIR,
)
from generator.generator_code_builder import validate_generator_source
from runner.generator_runner import GeneratorRunnerError, GeneratorRunResult, run_generator
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
from src.schema import SchemaValidationError, validate_generator_code, validate_test_data_spec


SYSTEM_PROMPT = """
You are a generator.py repair agent for an ACM/ICPC test data generation project.
Repair the provided Python generator so it compiles and creates valid .in files from the TestDataSpec.

Rules:
- Return JSON only.
- The top-level JSON object must contain exactly one key: "generator_code".
- "generator_code" must be the complete repaired Python source code, not a patch.
- Keep the generator self-contained and portable.
- Do not generate expected outputs, .out files, answer files, or solution code.
- The generator must accept an optional CLI argument for output directory.
- If no output directory is provided, default to "test_data".
- Use only Python standard library modules allowed by the project validator.
- Prefer minimal repairs over rewriting everything.
- Preserve deterministic randomness with a fixed seed.
- Generate large data algorithmically; do not embed huge literal inputs.
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous repair was invalid. Return one complete JSON object only.
The first character must be { and the last character must be }.
Return complete generator_code, not a diff.
""".strip()


def repair_and_run_generator(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    generator_path: Path = GENERATED_GENERATOR_PATH,
    output_dir: Path = TEST_DATA_DIR,
    max_attempts: int = GENERATOR_REPAIR_MAX_ATTEMPTS,
) -> GeneratorRunResult:
    normalized_spec = validate_test_data_spec(test_data_spec)
    generator_path = generator_path.resolve()
    output_dir = output_dir.resolve()

    last_error = ""
    for attempt in range(max_attempts + 1):
        try:
            return run_generator(generator_path, output_dir)
        except GeneratorRunnerError as exc:
            last_error = str(exc)
            if attempt >= max_attempts:
                raise GeneratorRunnerError(
                    f"Generator still failed after {max_attempts} repair attempt(s). Last error: {last_error}"
                ) from exc

            repaired_code = repair_generator_code(
                problem_info=problem_info,
                test_data_spec=normalized_spec,
                current_code=read_generator_code(generator_path),
                error_message=last_error,
                attempt=attempt + 1,
            )
            generator_path.parent.mkdir(parents=True, exist_ok=True)
            generator_path.write_text(repaired_code + "\n", encoding="utf-8")

    raise GeneratorRunnerError(f"Generator repair failed unexpectedly. Last error: {last_error}")


def repair_generator_code(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    current_code: str,
    error_message: str,
    attempt: int,
) -> str:
    if OpenAI is None:
        raise MissingDependencyError("The 'openai' package is not installed. Run: pip install -r requirements.txt")

    if not DEEPSEEK_API_KEY:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set. Set it in your environment before running this command."
        )

    user_prompt = build_repair_prompt(problem_info, test_data_spec, current_code, error_message, attempt)
    last_raw_output = ""
    last_validation_error = ""

    for retry in range(DEEPSEEK_MAX_RETRIES + 1):
        raw_output = request_repair_json(user_prompt, retry_note=last_validation_error if retry > 0 else "")
        last_raw_output = raw_output
        try:
            parsed = parse_json_output(raw_output)
            code = validate_generator_code(parsed)["generator_code"]
            validate_generator_source(code)
            return code
        except (ModelJSONParseError, SchemaValidationError) as exc:
            last_validation_error = str(exc)

    raise ModelJSONParseError(
        f"Repair agent returned invalid generator code after retries. Last error: {last_validation_error}",
        last_raw_output,
    )


def build_repair_prompt(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    current_code: str,
    error_message: str,
    attempt: int,
) -> str:
    return (
        f"repair_attempt: {attempt}\n\n"
        "problem_info:\n"
        f"{json.dumps(problem_info, ensure_ascii=False, indent=2)}\n\n"
        "test_data_spec:\n"
        f"{json.dumps(test_data_spec, ensure_ascii=False, indent=2)}\n\n"
        "generator_error:\n"
        f"{error_message}\n\n"
        "current_generator_code:\n"
        f"{current_code}"
    )


def request_repair_json(user_prompt: str, retry_note: str = "") -> str:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
    )

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
            max_tokens=GENERATOR_CODE_MAX_OUTPUT_TOKENS,
            extra_body={"thinking": {"type": DEEPSEEK_THINKING}},
        )
    except OpenAIError as exc:
        raise ModelRequestError(f"DeepSeek repair request failed: {exc}") from exc

    raw_output = (response.choices[0].message.content or "").strip()
    if not raw_output:
        response_id = getattr(response, "id", "unknown")
        finish_reason = response.choices[0].finish_reason or "unknown"
        raise ModelJSONParseError(
            "Repair agent returned an empty response. "
            f"model={DEEPSEEK_MODEL}, finish_reason={finish_reason}, response_id={response_id}.",
            raw_output,
        )
    return raw_output


def read_generator_code(generator_path: Path) -> str:
    if not generator_path.exists():
        return ""
    return generator_path.read_text(encoding="utf-8")
