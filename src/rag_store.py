from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_DIR,
    RAG_TOP_K,
)
from src.rag_splitter import KnowledgeChunk, load_markdown_chunks


class RAGError(Exception):
    """Base exception for RAG storage and retrieval errors."""


class RAGDependencyError(RAGError):
    """Raised when Chroma or the embedding package is not installed."""


def build_knowledge_base(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    persist_dir: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    reset: bool = False,
) -> int:
    chunks = load_markdown_chunks(knowledge_dir)
    if not chunks:
        raise RAGError(f"No Markdown chunks found in {knowledge_dir}")

    client = get_chroma_client(persist_dir)
    if reset:
        delete_collection_if_exists(client, collection_name)

    collection = get_collection(client, collection_name)
    upsert_chunks(collection, chunks)
    return len(chunks)


def retrieve_knowledge(
    query: str,
    top_k: int = RAG_TOP_K,
    persist_dir: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty.")

    collection = get_collection(get_chroma_client(persist_dir), collection_name)
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return normalize_query_result(result)


def build_query_from_problem_analysis(analysis: dict[str, Any]) -> str:
    parts = [
        str(analysis.get("title", "")),
        str(analysis.get("problem_type", "")),
        " ".join(str(tag) for tag in analysis.get("algorithm_tags", [])),
        str(analysis.get("constraints", "")),
        str(analysis.get("corner_cases", "")),
    ]
    return "\n".join(part for part in parts if part.strip())


def get_chroma_client(persist_dir: Path = CHROMA_DB_DIR) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RAGDependencyError("The 'chromadb' package is not installed. Run: pip install -r requirements.txt") from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_embedding_function() -> Any:
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RAGDependencyError(
            "SentenceTransformerEmbeddingFunction is unavailable. Run: pip install -r requirements.txt"
        ) from exc

    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        normalize_embeddings=True,
    )


def get_collection(client: Any, collection_name: str = CHROMA_COLLECTION_NAME) -> Any:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def delete_collection_if_exists(client: Any, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        return


def upsert_chunks(collection: Any, chunks: list[KnowledgeChunk], batch_size: int = 64) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.document for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )


def normalize_query_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    normalized: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        normalized.append(
            {
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return normalized
