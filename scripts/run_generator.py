from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GENERATED_GENERATOR_PATH, GENERATOR_RUN_TIMEOUT_SECONDS, TEST_DATA_DIR
from runner.generator_runner import GeneratorRunnerError, run_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run generator.py to produce test_data/*.in.")
    parser.add_argument("--generator", default=str(GENERATED_GENERATOR_PATH), help="Path to generator.py.")
    parser.add_argument("--output-dir", default=str(TEST_DATA_DIR), help="Directory for generated .in files.")
    parser.add_argument("--timeout", type=int, default=GENERATOR_RUN_TIMEOUT_SECONDS, help="Run timeout in seconds.")
    args = parser.parse_args()

    try:
        result = run_generator(Path(args.generator), Path(args.output_dir), args.timeout)
    except GeneratorRunnerError as exc:
        print(f"Generator error: {exc}")
        return

    print(f"Generated {len(result.input_files)} input files in {result.output_dir}:")
    for path in result.input_files:
        print(path)
    if result.stdout.strip():
        print("\nGenerator stdout:")
        print(result.stdout)


if __name__ == "__main__":
    main()
