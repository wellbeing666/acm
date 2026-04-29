from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GENERATED_GENERATOR_PATH
from generator.generator_code_builder import build_generator_code
from src.llm_parser import (
    MissingAPIKeyError,
    MissingDependencyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generator.py from problem_info and TestDataSpec.")
    parser.add_argument("--problem-info", required=True, help="Path to problem_info JSON.")
    parser.add_argument("--test-data-spec", required=True, help="Path to TestDataSpec JSON.")
    parser.add_argument("--output", default=str(GENERATED_GENERATOR_PATH), help="Output generator.py path.")
    args = parser.parse_args()

    try:
        problem_info = read_json_object(Path(args.problem_info), "problem_info")
        test_data_spec = read_json_object(Path(args.test_data_spec), "test_data_spec")
        output_path = build_generator_code(problem_info, test_data_spec, Path(args.output))
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

    print(f"Generator written to: {output_path}")


def read_json_object(path: Path, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return data


if __name__ == "__main__":
    main()
