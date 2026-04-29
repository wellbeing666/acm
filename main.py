import json
from pathlib import Path
from typing import Any

from config import APP_NAME
from runner.generator_runner import GeneratorRunnerError
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)
from src.pipeline import run_generation_pipeline
from src.rag_store import RAGError


def read_problem_statement() -> str:
    """Read a multiline ACM problem statement from the command line."""
    print(f"{APP_NAME} - ACM Test Case Generator")
    print("Paste the ACM problem statement. Press Enter on an empty line to finish:")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main() -> None:
    problem_statement = read_problem_statement()

    if not problem_statement:
        print("No problem statement received.")
        return

    try:
        result = run_generation_pipeline(problem_statement, on_stage=print_stage_update)
    except MissingDependencyError as exc:
        print(f"Dependency error: {exc}")
        return
    except MissingAPIKeyError as exc:
        print(f"Configuration error: {exc}")
        return
    except ModelRequestError as exc:
        print(f"Model request error: {exc}")
        return
    except ModelJSONParseError as exc:
        print(f"JSON parse error: {exc}")
        if exc.raw_output:
            print("\nRaw model output:")
            print(exc.raw_output)
        return
    except ModelSchemaError as exc:
        print(f"Schema validation error: {exc}")
        return
    except RAGError as exc:
        print(f"RAG error: {exc}")
        print("Build the knowledge base with: python scripts/ingest_knowledge.py --reset")
        return
    except GeneratorRunnerError as exc:
        print(f"Generator error: {exc}")
        return
    except ValueError as exc:
        print(f"Input error: {exc}")
        return

    print(f"\nGenerated {len(result.input_files)} input files:")
    for path in result.input_files:
        print(path)
    print(f"\nGenerator: {result.generator_path}")
    print(f"Test data dir: {result.test_data_dir}")
    if result.agent_trace_path:
        print(f"Agent trace: {result.agent_trace_path}")


def print_stage_update(stage: str, payload: dict[str, Any]) -> None:
    message = payload.get("message", stage)
    print(f"\n[{stage}] {message}")

    if "problem_info" in payload:
        print(json.dumps(payload["problem_info"], ensure_ascii=False, indent=2))
    if "test_plan" in payload:
        print(json.dumps(payload["test_plan"], ensure_ascii=False, indent=2))
    if "test_data_spec" in payload:
        print(json.dumps(payload["test_data_spec"], ensure_ascii=False, indent=2))
    if "generator_path" in payload:
        print(f"Generator path: {format_path(payload['generator_path'])}")
    if "test_data_dir" in payload:
        print(f"Test data dir: {format_path(payload['test_data_dir'])}")
    if "input_files" in payload:
        for path in payload["input_files"]:
            print(format_path(path))
    if "agent_trace_path" in payload and payload["agent_trace_path"]:
        print(f"Agent trace: {format_path(payload['agent_trace_path'])}")


def format_path(path: Any) -> str:
    return str(path) if isinstance(path, Path) else str(path)


if __name__ == "__main__":
    main()
