from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.test_spec_generator import generate_test_data_spec
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TestDataSpec from problem_info, test_plan, and context.")
    parser.add_argument("--problem-info", required=True, help="Path to problem_info JSON.")
    parser.add_argument("--test-plan", required=True, help="Path to test_plan JSON.")
    parser.add_argument("--retrieved-context", required=True, help="Path to retrieved_context JSON or text.")
    args = parser.parse_args()

    try:
        problem_info = read_json_object(Path(args.problem_info), "problem_info")
        test_plan = read_json_object(Path(args.test_plan), "test_plan")
        retrieved_context = read_json_or_text(Path(args.retrieved_context))
        spec = generate_test_data_spec(problem_info, test_plan, retrieved_context)
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

    print(json.dumps(spec, ensure_ascii=False, indent=2))


def read_json_object(path: Path, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return data


def read_json_or_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


if __name__ == "__main__":
    main()
