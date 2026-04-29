from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from runner.generator_runner import GeneratorRunnerError
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)
from src.pipeline import make_run_id, resolve_output_paths, run_generation_pipeline
from src.rag_store import RAGError
from src.workflow_publisher import SafeWorkflowPublisher, publish_generated_artifacts


app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/generate")
def generate() -> Any:
    payload = request.get_json(silent=True) or {}
    problem_statement = build_problem_statement(payload)
    session_id = get_session_id(payload)
    fallback_generator_path, fallback_test_data_dir = resolve_output_paths(session_id, isolated_outputs=True)

    if not isinstance(problem_statement, str) or not problem_statement.strip():
        return jsonify(make_api_response(False, fallback_generator_path, [])), 400

    publisher = SafeWorkflowPublisher(session_id)
    try:
        publisher.info("workflow started")
        result = run_generation_pipeline(
            problem_statement=problem_statement,
            run_id=session_id,
            isolated_outputs=True,
            on_stage=publisher.publish_stage,
        )
        publish_generated_artifacts(publisher, result.generator_path, result.input_files)
        publisher.info("workflow completed")
    except (
        MissingDependencyError,
        MissingAPIKeyError,
        ModelRequestError,
        ModelJSONParseError,
        ModelSchemaError,
        RAGError,
        GeneratorRunnerError,
        ValueError,
    ) as exc:
        publisher.error("workflow failed", error_type=exc.__class__.__name__, error=str(exc))
        publisher.close()
        return jsonify(
            make_api_response(False, fallback_generator_path, collect_input_files(fallback_test_data_dir))
        ), 500
    except Exception as exc:
        publisher.error("workflow failed", error_type=exc.__class__.__name__, error=str(exc))
        publisher.close()
        return jsonify(
            make_api_response(False, fallback_generator_path, collect_input_files(fallback_test_data_dir))
        ), 500
    finally:
        publisher.close()

    return jsonify(make_api_response(True, result.generator_path, result.input_files))


def build_problem_statement(payload: dict[str, Any]) -> str:
    problem = payload.get("problem")
    user_input = payload.get("input")

    if problem is None:
        problem = request.form.get("problem")
    if user_input is None:
        user_input = request.form.get("input")

    if not isinstance(problem, str):
        problem = ""
    if not isinstance(user_input, str):
        user_input = ""

    if not problem.strip():
        return ""

    parts = [f"Problem Statement:\n{problem.strip()}"]
    if user_input.strip():
        parts.append(f"User Input / Extra Requirements:\n{user_input.strip()}")

    return "\n\n".join(part for part in parts if part.strip())


def get_session_id(payload: dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    if session_id is None:
        session_id = request.form.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = make_run_id()
    return session_id.strip()


def make_api_response(ok: bool, generator_path: Path, input_files: list[Path]) -> dict[str, Any]:
    return {
        "ok": ok,
        "generator_path": str(generator_path.resolve()),
        "input_files": [str(path.resolve()) for path in input_files],
    }


def collect_input_files(test_data_dir: Path) -> list[Path]:
    if not test_data_dir.exists():
        return []
    return sorted(test_data_dir.glob("*.in"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
