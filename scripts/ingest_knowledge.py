from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, KNOWLEDGE_DIR
from src.rag_store import RAGError, build_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persistent Chroma knowledge base.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    args = parser.parse_args()

    try:
        count = build_knowledge_base(reset=args.reset)
    except RAGError as exc:
        print(f"RAG ingest error: {exc}")
        return

    print("Knowledge base built successfully.")
    print(f"Knowledge dir: {KNOWLEDGE_DIR}")
    print(f"Chroma dir: {CHROMA_DB_DIR}")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Chunks indexed: {count}")


if __name__ == "__main__":
    main()
