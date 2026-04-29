from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.generator_repair_agent import repair_and_run_generator_with_trace
from config import GENERATED_GENERATOR_PATH, GENERATOR_REPAIR_MAX_ATTEMPTS, TEST_DATA_DIR
from runner.generator_runner import GeneratorRunnerError
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generator.py and repair it with an LLM agent if it fails.")
    parser.add_argument("--problem-info", required=True, help="Path to problem_info JSON.")
    parser.add_argument("--test-data-spec", required=True, help="Path to TestDataSpec JSON.")
    parser.add_argument("--generator", default=str(GENERATED_GENERATOR_PATH), help="Path to generator.py.")
    parser.add_argument("--output-dir", default=str(TEST_DATA_DIR), help="Directory for generated .in files.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=GENERATOR_REPAIR_MAX_ATTEMPTS,
        help="Maximum LLM repair attempts.",
    )
    args = parser.parse_args()

    try:
        problem_info = read_json_object(Path(args.problem_info), "problem_info")
        test_data_spec = read_json_object(Path(args.test_data_spec), "test_data_spec")
        agent_result = repair_and_run_generator_with_trace(
            problem_info=problem_info,
            test_data_spec=test_data_spec,
            generator_path=Path(args.generator),
            output_dir=Path(args.output_dir),
            max_attempts=args.max_attempts,
        )
        result = agent_result.run_result
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Input error: {exc}")
        return
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
    except GeneratorRunnerError as exc:
        print(f"Generator error: {exc}")
        return

    print(f"Generated {len(result.input_files)} input files in {result.output_dir}:")
    for path in result.input_files:
        print(path)
    if agent_result.trace_path:
        print(f"\nAgent trace: {agent_result.trace_path}")


def read_json_object(path: Path, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return data


if __name__ == "__main__":
    main()
