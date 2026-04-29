import json

from config import APP_NAME, RAG_TOP_K
from src.llm_parser import (
    MissingDependencyError,
    MissingAPIKeyError,
    ModelJSONParseError,
    ModelRequestError,
    ModelSchemaError,
    analyze_problem_statement,
)
from src.rag_store import RAGError, build_query_from_problem_analysis, retrieve_knowledge
from src.test_plan_generator import generate_test_plan
from generator.generator_code_builder import build_generator_code
from generator.test_spec_generator import generate_test_data_spec
from agent.generator_repair_agent import repair_and_run_generator
from runner.generator_runner import GeneratorRunnerError


def read_problem_statement() -> str:
    """Read a multiline ACM problem statement from the command line."""
    print(f"{APP_NAME} - ACM Test Case Generator")
    print("Paste the ACM problem statement. Press Enter on an empty line to finish:")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main() -> None:
    problem_statement = read_problem_statement()

    if not problem_statement:
        print("No problem statement received.")
        return

    print("\nAnalyzing problem statement with the model...")

    try:
        analysis = analyze_problem_statement(problem_statement)
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

    print("\nStructured JSON:")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    try:
        query = build_query_from_problem_analysis(analysis)
        knowledge_chunks = retrieve_knowledge(query, top_k=RAG_TOP_K)
    except RAGError as exc:
        print(f"\nRAG retrieval skipped: {exc}")
        print("Build the knowledge base with: python scripts/ingest_knowledge.py --reset")
        return

    if not knowledge_chunks:
        print("\nNo RAG knowledge chunks found. Build the knowledge base first:")
        print("python scripts/ingest_knowledge.py --reset")
        return

    print("\nRetrieved RAG knowledge:")
    for index, item in enumerate(knowledge_chunks, start=1):
        metadata = item["metadata"]
        print(f"\n[{index}] {metadata.get('source')} / {metadata.get('section')}")
        print(item["document"])

    print("\nGenerating test plan with the model...")
    try:
        test_plan = generate_test_plan(analysis, knowledge_chunks)
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

    print("\nTest plan JSON:")
    print(json.dumps(test_plan, ensure_ascii=False, indent=2))

    print("\nGenerating TestDataSpec with the model...")
    try:
        test_data_spec = generate_test_data_spec(analysis, test_plan, knowledge_chunks)
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

    print("\nTestDataSpec JSON:")
    print(json.dumps(test_data_spec, ensure_ascii=False, indent=2))

    print("\nBuilding generator.py with the model...")
    try:
        generator_path = build_generator_code(analysis, test_data_spec)
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

    print(f"\nGenerator written to: {generator_path}")

    print("\nRunning generator.py with repair agent...")
    try:
        result = repair_and_run_generator(analysis, test_data_spec, generator_path)
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

    print(f"\nGenerated {len(result.input_files)} input files:")
    for path in result.input_files:
        print(path)


if __name__ == "__main__":
    main()
