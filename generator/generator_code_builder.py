from __future__ import annotations

import ast
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
    GENERATED_GENERATOR_PATH,
    TEST_DATA_DIR,
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
from src.schema import SchemaValidationError, validate_generator_code, validate_test_data_spec


SYSTEM_PROMPT = """
You are an ACM/ICPC Python test data generator engineer.
Create a self-contained Python generator.py from problem_info and TestDataSpec.

Rules:
- Return JSON only.
- Do not include markdown fences, comments outside the code string, or explanatory text.
- The top-level JSON object must contain exactly one key: "generator_code".
- "generator_code" must be raw Python source code as a JSON string.
- The generator must create input files only, with names like TC001_01.in under the output directory.
- Do not generate expected outputs, .out files, answer files, or solution code.
- The generated Python code must accept an optional CLI argument: output directory.
- If no output directory is provided, default to "test_data".
- Use only Python standard library modules: pathlib, random, argparse, itertools, math, collections.
- Use deterministic randomness with a fixed seed.
- Validate or assert key constraints before writing each file.
- Keep generated code portable on Windows and Linux.
- Avoid writing very large literal strings in source code; generate large data algorithmically.

Required JSON shape:
{
  "generator_code": "from pathlib import Path\\n..."
}
""".strip()

RETRY_PROMPT_SUFFIX = """

Important: Your previous response was empty or invalid. Output one complete valid JSON object only.
The first character must be { and the last character must be }.
The code must generate only .in files. Do not create .out files, answer files, or expected_output fields.
""".strip()

ALLOWED_IMPORTS = {
    "argparse",
    "collections",
    "itertools",
    "math",
    "pathlib",
    "random",
    "sys",
}
FORBIDDEN_TOKENS = (
    "subprocess",
    "socket",
    "requests",
    "shutil.rmtree",
    "os.system",
    "eval(",
    "exec(",
    "__import__",
    "expected_output",
)


def build_generator_code(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    output_path: Path = GENERATED_GENERATOR_PATH,
) -> Path:
    if OpenAI is None:
        raise MissingDependencyError("The 'openai' package is not installed. Run: pip install -r requirements.txt")

    if not DEEPSEEK_API_KEY:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set. Set it in your environment before running this command."
        )

    normalized_spec = validate_test_data_spec(test_data_spec)
    user_prompt = build_generator_code_prompt(problem_info, normalized_spec)
    last_raw_output = ""
    last_error = ""

    for attempt in range(DEEPSEEK_MAX_RETRIES + 1):
        raw_output = request_json_from_model(user_prompt, retry_note=last_error if attempt > 0 else "")
        last_raw_output = raw_output
        try:
            parsed = parse_json_output(raw_output)
            code = validate_generator_code(parsed)["generator_code"]
            validate_generator_source(code)
            break
        except (ModelJSONParseError, SchemaValidationError) as exc:
            last_error = str(exc)
    else:
        raise ModelJSONParseError(
            f"Model returned invalid generator code JSON after retries. Last error: {last_error}",
            last_raw_output,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code + "\n", encoding="utf-8")
    return output_path


def build_generator_code_prompt(problem_info: dict[str, Any], test_data_spec: dict[str, Any]) -> str:
    return (
        "problem_info:\n"
        f"{json.dumps(problem_info, ensure_ascii=False, indent=2)}\n\n"
        "test_data_spec:\n"
        f"{json.dumps(test_data_spec, ensure_ascii=False, indent=2)}\n\n"
        f"output_directory_default: {TEST_DATA_DIR.as_posix()}\n"
    )


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
            max_tokens=GENERATOR_CODE_MAX_OUTPUT_TOKENS,
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
        "Model returned an empty generator code response after retries. "
        f"model={DEEPSEEK_MODEL}, finish_reason={last_finish_reason}, response_id={last_response_id}.",
        raw_output,
    )


def validate_generator_source(code: str) -> None:
    lowered = code.lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            raise SchemaValidationError(f"Generator code contains forbidden token: {token}")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SchemaValidationError(f"Generator code is not valid Python: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise SchemaValidationError(f"Generator code imports forbidden module: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise SchemaValidationError(f"Generator code imports forbidden module: {node.module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            validate_string_literal(node.value)


def validate_string_literal(value: str) -> None:
    lowered = value.lower()
    if lowered.strip().endswith(".out"):
        raise SchemaValidationError("Generator code must not create .out files.")
    if ".out/" in lowered or ".out\\" in lowered:
        raise SchemaValidationError("Generator code must not create .out files.")
