from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from agent.generator_repair_agent import repair_and_run_generator_with_trace
from config import GENERATED_GENERATOR_PATH, GENERATED_RUNS_DIR, RAG_TOP_K, TEST_DATA_DIR
from generator.generator_code_builder import build_generator_code
from generator.test_spec_generator import generate_test_data_spec
from src.llm_parser import analyze_problem_statement
from src.rag_store import build_query_from_problem_analysis, retrieve_knowledge
from src.test_plan_generator import generate_test_plan


PipelineCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PipelineResult:
    problem_info: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    test_plan: dict[str, Any]
    test_data_spec: dict[str, Any]
    generator_path: Path
    test_data_dir: Path
    input_files: list[Path]
    agent_trace_path: Path | None

    def to_response_dict(self) -> dict[str, Any]:
        generator_path = self.generator_path.resolve()
        test_data_dir = self.test_data_dir.resolve()
        input_files = [path.resolve() for path in self.input_files]
        agent_trace_path = self.agent_trace_path.resolve() if self.agent_trace_path else None

        return {
            "problem_info": self.problem_info,
            "generator_path": str(generator_path),
            "generator_dir": str(generator_path.parent),
            "test_data_dir": str(test_data_dir),
            "input_files": [str(path) for path in input_files],
            "agent_trace_path": str(agent_trace_path) if agent_trace_path else None,
            "counts": {
                "retrieved_context": len(self.retrieved_context),
                "test_plan": len(self.test_plan.get("test_plan", [])),
                "test_data_spec_cases": len(self.test_data_spec.get("test_data_spec", {}).get("cases", [])),
                "input_files": len(self.input_files),
            },
        }


def run_generation_pipeline(
    problem_statement: str,
    run_id: str | None = None,
    isolated_outputs: bool = False,
    on_stage: PipelineCallback | None = None,
) -> PipelineResult:
    if not problem_statement.strip():
        raise ValueError("problem_statement must not be empty.")

    emit(on_stage, "analyze_problem", message="Analyzing problem statement.")
    problem_info = analyze_problem_statement(problem_statement)
    emit(on_stage, "analyze_problem", message="Problem statement parsed.", problem_info=problem_info)

    emit(on_stage, "retrieve_knowledge", message="Retrieving RAG knowledge.")
    query = build_query_from_problem_analysis(problem_info)
    retrieved_context = retrieve_knowledge(query, top_k=RAG_TOP_K)
    if not retrieved_context:
        raise ValueError("No RAG knowledge chunks found. Build the knowledge base first.")
    emit(on_stage, "retrieve_knowledge", message="RAG knowledge retrieved.", count=len(retrieved_context))

    emit(on_stage, "generate_test_plan", message="Generating test plan.")
    test_plan = generate_test_plan(problem_info, retrieved_context)
    emit(on_stage, "generate_test_plan", message="Test plan generated.", test_plan=test_plan)

    emit(on_stage, "generate_test_data_spec", message="Generating TestDataSpec.")
    test_data_spec = generate_test_data_spec(problem_info, test_plan, retrieved_context)
    emit(on_stage, "generate_test_data_spec", message="TestDataSpec generated.", test_data_spec=test_data_spec)

    generator_path, test_data_dir = resolve_output_paths(run_id, isolated_outputs)

    emit(on_stage, "build_generator", message="Building generator.py.", generator_path=generator_path)
    generator_path = build_generator_code(problem_info, test_data_spec, generator_path)
    emit(on_stage, "build_generator", message="generator.py written.", generator_path=generator_path)

    emit(on_stage, "run_generator", message="Running generator.py with repair agent.", test_data_dir=test_data_dir)
    agent_result = repair_and_run_generator_with_trace(problem_info, test_data_spec, generator_path, test_data_dir)
    run_result = agent_result.run_result
    emit(
        on_stage,
        "run_generator",
        message="Input files generated.",
        input_files=run_result.input_files,
        agent_trace_path=agent_result.trace_path,
    )

    return PipelineResult(
        problem_info=problem_info,
        retrieved_context=retrieved_context,
        test_plan=test_plan,
        test_data_spec=test_data_spec,
        generator_path=generator_path,
        test_data_dir=run_result.output_dir,
        input_files=run_result.input_files,
        agent_trace_path=agent_result.trace_path,
    )


def resolve_output_paths(run_id: str | None, isolated_outputs: bool) -> tuple[Path, Path]:
    if not isolated_outputs:
        return GENERATED_GENERATOR_PATH, TEST_DATA_DIR

    safe_run_id = run_id or make_run_id()
    return GENERATED_RUNS_DIR / safe_run_id / "generator.py", TEST_DATA_DIR / safe_run_id


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


def emit(callback: PipelineCallback | None, stage: str, **payload: Any) -> None:
    if callback:
        callback(stage, payload)
