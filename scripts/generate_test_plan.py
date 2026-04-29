from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)
from src.test_plan_generator import generate_test_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a JSON test plan from problem_info and retrieved_context.")
    parser.add_argument("--problem-info", required=True, help="Path to a JSON file containing problem_info.")
    parser.add_argument(
        "--retrieved-context",
        required=True,
        help="Path to retrieved_context as JSON or plain text.",
    )
    args = parser.parse_args()

    try:
        problem_info = read_json(Path(args.problem_info))
        retrieved_context = read_json_or_text(Path(args.retrieved_context))
        test_plan = generate_test_plan(problem_info, retrieved_context)
    except OSError as exc:
        print(f"File error: {exc}")
        return
    except json.JSONDecodeError as exc:
        print(f"JSON file parse error: {exc}")
        return
    except ValueError as exc:
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

    print(json.dumps(test_plan, ensure_ascii=False, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def read_json_or_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


if __name__ == "__main__":
    main()
