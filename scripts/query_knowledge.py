from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAG_TOP_K
from src.rag_store import RAGError, retrieve_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Chroma knowledge base.")
    parser.add_argument("query", help="Natural-language query or parsed problem summary.")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    try:
        results = retrieve_knowledge(args.query, top_k=args.top_k)
    except RAGError as exc:
        print(f"RAG query error: {exc}")
        return

    for index, item in enumerate(results, start=1):
        metadata = item["metadata"]
        print(f"\n[{index}] distance={item['distance']}")
        print(f"source={metadata.get('source')} section={metadata.get('section')}")
        print(item["document"])


if __name__ == "__main__":
    main()
