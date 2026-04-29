import unittest
from pathlib import Path

from app import app, build_problem_statement, make_api_response


class FlaskAppTests(unittest.TestCase):
    def test_health(self) -> None:
        client = app.test_client()
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_generate_requires_problem_statement(self) -> None:
        client = app.test_client()
        response = client.post("/generate", json={})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(set(data.keys()), {"ok", "generator_path", "input_files"})
        self.assertFalse(data["ok"])

    def test_build_problem_statement_merges_problem_and_input(self) -> None:
        with app.test_request_context(
            "/generate",
            method="POST",
            json={"problem": "problem text", "input": "extra text"},
        ):
            text = build_problem_statement({"problem": "problem text", "input": "extra text"})

        self.assertIn("Problem Statement", text)
        self.assertIn("problem text", text)
        self.assertIn("User Input / Extra Requirements", text)
        self.assertIn("extra text", text)

    def test_make_api_response_shape_and_absolute_paths(self) -> None:
        data = make_api_response(
            ok=True,
            generator_path=Path("data/generated/runs/demo/generator.py"),
            input_files=[Path("test_data/demo/TC001_01.in")],
        )

        self.assertEqual(set(data.keys()), {"ok", "generator_path", "input_files"})
        self.assertTrue(data["ok"])
        self.assertTrue(Path(data["generator_path"]).is_absolute())
        self.assertTrue(Path(data["input_files"][0]).is_absolute())

    def test_generate_uses_request_session_id_for_paths_but_not_response_field(self) -> None:
        client = app.test_client()
        response = client.post("/generate", json={"session_id": "session-1"})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertNotIn("session_id", data)
        self.assertIn("session-1", data["generator_path"])


if __name__ == "__main__":
    unittest.main()
