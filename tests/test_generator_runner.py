import unittest
from pathlib import Path

from runner.generator_runner import run_generator


class GeneratorRunnerTests(unittest.TestCase):
    def test_run_generator_creates_input_files(self) -> None:
        root = Path(__file__).parent / "runtime"
        generator_path = root / "generator.py"
        output_dir = root / "test_data"
        generator_path.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "def main():\n"
            "    out = Path(sys.argv[1])\n"
            "    out.mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'TC001_01.in').write_text('1\\n', encoding='utf-8')\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
            encoding="utf-8",
        )

        result = run_generator(generator_path, output_dir, timeout_seconds=5)

        self.assertEqual(len(result.input_files), 1)
        self.assertEqual(result.input_files[0].name, "TC001_01.in")


if __name__ == "__main__":
    unittest.main()
