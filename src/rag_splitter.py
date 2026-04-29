from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    document: str
    metadata: dict[str, str | int | float | bool | None]


def load_markdown_chunks(
    knowledge_dir: Path,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for markdown_path in sorted(knowledge_dir.glob("*.md")):
        chunks.extend(split_markdown_file(markdown_path, knowledge_dir, chunk_size, chunk_overlap))
    return chunks


def split_markdown_file(
    markdown_path: Path,
    knowledge_dir: Path,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> list[KnowledgeChunk]:
    text = markdown_path.read_text(encoding="utf-8")
    front_matter, body = parse_front_matter(text)
    sections = split_by_h2(body)

    chunks: list[KnowledgeChunk] = []
    relative_source = markdown_path.relative_to(knowledge_dir).as_posix()

    for section_index, (section_title, section_text) in enumerate(sections):
        for chunk_index, chunk_text in enumerate(split_text(section_text, chunk_size, chunk_overlap)):
            normalized = chunk_text.strip()
            if not normalized:
                continue

            chunk_id = make_chunk_id(relative_source, section_index, chunk_index, normalized)
            metadata = build_metadata(front_matter, relative_source, section_title, section_index, chunk_index)
            chunks.append(KnowledgeChunk(id=chunk_id, document=normalized, metadata=metadata))

    return chunks


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value.startswith("[") and raw_value.endswith("]"):
            values = [item.strip().strip("\"'") for item in raw_value[1:-1].split(",") if item.strip()]
            metadata[key] = values
        else:
            metadata[key] = raw_value.strip("\"'")

    return metadata, text[match.end() :]


def split_by_h2(markdown_body: str) -> list[tuple[str, str]]:
    matches = list(H2_RE.finditer(markdown_body))
    if not matches:
        title = first_heading(markdown_body) or "document"
        return [(title, markdown_body)]

    sections: list[tuple[str, str]] = []
    prefix = markdown_body[: matches[0].start()].strip()
    if prefix:
        sections.append((first_heading(prefix) or "overview", prefix))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_body)
        sections.append((match.group(1).strip(), markdown_body[start:end].strip()))

    return sections


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_text(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            current = with_overlap(current, chunk_overlap)
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def with_overlap(text: str, chunk_overlap: int) -> str:
    if chunk_overlap == 0:
        return ""
    return text[-chunk_overlap:].strip()


def build_metadata(
    front_matter: dict[str, Any],
    source: str,
    section: str,
    section_index: int,
    chunk_index: int,
) -> dict[str, str | int | float | bool | None]:
    tags = front_matter.get("algorithm_tags", [])
    if isinstance(tags, list):
        tag_text = ", ".join(str(tag) for tag in tags)
    else:
        tag_text = str(tags)

    return {
        "source": source,
        "title": str(front_matter.get("title", "")),
        "problem_type": str(front_matter.get("problem_type", "")),
        "algorithm_tags": tag_text,
        "section": section,
        "section_index": section_index,
        "chunk_index": chunk_index,
    }


def make_chunk_id(source: str, section_index: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{section_index}:{chunk_index}:{digest}"
