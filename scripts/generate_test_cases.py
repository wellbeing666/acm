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
from src.test_case_generator import generate_test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concrete ACM test cases from problem_info and test_plan.")
    parser.add_argument("--problem-info", required=True, help="Path to a JSON file containing problem_info.")
    parser.add_argument("--test-plan", required=True, help="Path to a JSON file containing test_plan.")
    args = parser.parse_args()

    try:
        problem_info = read_json_object(Path(args.problem_info), "problem_info")
        test_plan = read_json(Path(args.test_plan))
        test_cases = generate_test_cases(problem_info, test_plan)
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

    print(json.dumps(test_cases, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_object(path: Path, name: str) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return data


if __name__ == "__main__":
    main()
