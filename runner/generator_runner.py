from __future__ import annotations

import py_compile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config import GENERATED_GENERATOR_PATH, GENERATOR_RUN_TIMEOUT_SECONDS, TEST_DATA_DIR


class GeneratorRunnerError(Exception):
    """Raised when generator.py cannot be compiled or executed."""


@dataclass(frozen=True)
class GeneratorRunResult:
    generator_path: Path
    output_dir: Path
    input_files: list[Path]
    stdout: str
    stderr: str


def compile_generator(generator_path: Path = GENERATED_GENERATOR_PATH) -> None:
    generator_path = generator_path.resolve()
    if not generator_path.exists():
        raise GeneratorRunnerError(f"Generator file does not exist: {generator_path}")

    try:
        py_compile.compile(str(generator_path), doraise=True)
    except py_compile.PyCompileError as exc:
        raise GeneratorRunnerError(f"Generator compile failed: {exc.msg}") from exc


def run_generator(
    generator_path: Path = GENERATED_GENERATOR_PATH,
    output_dir: Path = TEST_DATA_DIR,
    timeout_seconds: int = GENERATOR_RUN_TIMEOUT_SECONDS,
) -> GeneratorRunResult:
    generator_path = generator_path.resolve()
    output_dir = output_dir.resolve()
    compile_generator(generator_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, str(generator_path), str(output_dir)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(generator_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GeneratorRunnerError(f"Generator timed out after {timeout_seconds} seconds.") from exc

    if completed.returncode != 0:
        raise GeneratorRunnerError(
            "Generator run failed with exit code "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    input_files = sorted(output_dir.glob("*.in"))
    if not input_files:
        raise GeneratorRunnerError(f"Generator did not create any .in files in {output_dir}")

    return GeneratorRunResult(
        generator_path=generator_path,
        output_dir=output_dir,
        input_files=input_files,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
