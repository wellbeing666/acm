from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    AGENT_TRACE_DIR,
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
from agent.runtime import AgentTool, AgentTrace
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


@dataclass(frozen=True)
class GeneratorRepairAgentResult:
    run_result: GeneratorRunResult
    trace: AgentTrace
    trace_path: Path | None = None


class GeneratorRunTool(AgentTool):
    name = "generator.run"

    def run(self, generator_path: Path, output_dir: Path) -> GeneratorRunResult:
        return run_generator(generator_path, output_dir)


class GeneratorRepairTool(AgentTool):
    name = "generator.repair"

    def run(
        self,
        problem_info: dict[str, Any],
        test_data_spec: dict[str, Any],
        current_code: str,
        error_message: str,
        attempt: int,
    ) -> str:
        return repair_generator_code(problem_info, test_data_spec, current_code, error_message, attempt)


class GeneratorWriteTool(AgentTool):
    name = "generator.write"

    def run(self, generator_path: Path, code: str) -> None:
        generator_path.parent.mkdir(parents=True, exist_ok=True)
        generator_path.write_text(code + "\n", encoding="utf-8")


class GeneratorRepairAgent:
    def __init__(
        self,
        problem_info: dict[str, Any],
        test_data_spec: dict[str, Any],
        generator_path: Path = GENERATED_GENERATOR_PATH,
        output_dir: Path = TEST_DATA_DIR,
        max_attempts: int = GENERATOR_REPAIR_MAX_ATTEMPTS,
        trace_dir: Path = AGENT_TRACE_DIR,
    ) -> None:
        self.problem_info = problem_info
        self.test_data_spec = validate_test_data_spec(test_data_spec)
        self.generator_path = generator_path.resolve()
        self.output_dir = output_dir.resolve()
        self.max_attempts = max_attempts
        self.trace_dir = trace_dir
        self.trace = AgentTrace(agent_name="GeneratorRepairAgent")
        self.run_tool = GeneratorRunTool()
        self.repair_tool = GeneratorRepairTool()
        self.write_tool = GeneratorWriteTool()

    def run(self, save_trace: bool = True) -> GeneratorRepairAgentResult:
        self.trace.add(
            phase="start",
            status="ok",
            message="Starting generator repair loop.",
            generator_path=self.generator_path,
            output_dir=self.output_dir,
            max_attempts=self.max_attempts,
        )

        last_error = ""
        for attempt in range(self.max_attempts + 1):
            self.trace.add(
                phase="observe",
                status="running",
                message="Compiling and running generator.",
                attempt=attempt,
                tool=self.run_tool.describe(),
            )
            try:
                run_result = self.run_tool.run(self.generator_path, self.output_dir)
                self.trace.add(
                    phase="verify",
                    status="success",
                    message="Generator produced input files.",
                    files=run_result.input_files,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                )
                trace_path = self.trace.save(self.trace_dir, "generator_repair") if save_trace else None
                return GeneratorRepairAgentResult(run_result=run_result, trace=self.trace, trace_path=trace_path)
            except GeneratorRunnerError as exc:
                last_error = str(exc)
                self.trace.add(
                    phase="observe",
                    status="failed",
                    message="Generator failed.",
                    attempt=attempt,
                    error=last_error,
                )

                if attempt >= self.max_attempts:
                    self.trace.add(
                        phase="stop",
                        status="failed",
                        message="Repair budget exhausted.",
                        max_attempts=self.max_attempts,
                        error=last_error,
                    )
                    if save_trace:
                        self.trace.save(self.trace_dir, "generator_repair_failed")
                    raise GeneratorRunnerError(
                        f"Generator still failed after {self.max_attempts} repair attempt(s). Last error: {last_error}"
                    ) from exc

                self.repair_once(error_message=last_error, attempt=attempt + 1)

        raise GeneratorRunnerError(f"Generator repair failed unexpectedly. Last error: {last_error}")

    def repair_once(self, error_message: str, attempt: int) -> None:
        current_code = read_generator_code(self.generator_path)
        self.trace.add(
            phase="plan",
            status="ok",
            message="Preparing repair prompt from error and current code.",
            attempt=attempt,
            error=error_message,
            current_code_length=len(current_code),
        )
        repaired_code = self.repair_tool.run(
            problem_info=self.problem_info,
            test_data_spec=self.test_data_spec,
            current_code=current_code,
            error_message=error_message,
            attempt=attempt,
        )
        self.trace.add(
            phase="act",
            status="ok",
            message="Repair model returned validated generator code.",
            attempt=attempt,
            repaired_code_length=len(repaired_code),
            tool=self.repair_tool.describe(),
        )
        self.write_tool.run(self.generator_path, repaired_code)
        self.trace.add(
            phase="act",
            status="ok",
            message="Repaired generator code written to disk.",
            attempt=attempt,
            generator_path=self.generator_path,
            tool=self.write_tool.describe(),
        )


def repair_and_run_generator(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    generator_path: Path = GENERATED_GENERATOR_PATH,
    output_dir: Path = TEST_DATA_DIR,
    max_attempts: int = GENERATOR_REPAIR_MAX_ATTEMPTS,
) -> GeneratorRunResult:
    return repair_and_run_generator_with_trace(
        problem_info=problem_info,
        test_data_spec=test_data_spec,
        generator_path=generator_path,
        output_dir=output_dir,
        max_attempts=max_attempts,
    ).run_result


def repair_and_run_generator_with_trace(
    problem_info: dict[str, Any],
    test_data_spec: dict[str, Any],
    generator_path: Path = GENERATED_GENERATOR_PATH,
    output_dir: Path = TEST_DATA_DIR,
    max_attempts: int = GENERATOR_REPAIR_MAX_ATTEMPTS,
) -> GeneratorRepairAgentResult:
    agent = GeneratorRepairAgent(
        problem_info=problem_info,
        test_data_spec=test_data_spec,
        generator_path=generator_path,
        output_dir=output_dir,
        max_attempts=max_attempts,
    )
    return agent.run(save_trace=True)


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
