from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    RABBITMQ_ENABLED,
    RABBITMQ_EXCHANGE,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_REQUIRED,
    RABBITMQ_USER,
)

try:
    import pika
    from pika import BlockingConnection, PlainCredentials
except ImportError:  # pragma: no cover - exercised only when dependencies are missing.
    pika = None
    BlockingConnection = None
    PlainCredentials = None


class RabbitMQPublishError(RuntimeError):
    """Raised when RabbitMQ publishing is required but unavailable."""


class AgentPublisher:
    """RabbitMQ event publisher compatible with the provided agent_publisher.py message shape."""

    def __init__(
        self,
        session_id: str,
        host: str = RABBITMQ_HOST,
        port: int = RABBITMQ_PORT,
        user: str = RABBITMQ_USER,
        password: str = RABBITMQ_PASSWORD,
        exchange: str = RABBITMQ_EXCHANGE,
    ) -> None:
        self.session_id = session_id
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.exchange = exchange
        self._event_id = 0
        self._lock = threading.Lock()
        self._connection: Any = None
        self._channel: Any = None

    def publish(
        self,
        event_type: str,
        content: str = "",
        file_type: str | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        event_id = self._next_event_id()
        message: dict[str, Any] = {
            "session_id": self.session_id,
            "event_id": event_id,
            "event_type": event_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if event_type == "GENERATED":
            message["data"] = {
                "file_type": file_type or "",
                "file_path": file_path or "",
            }

        if kwargs:
            message.setdefault("data", {}).update(to_jsonable(kwargs))

        self._connect()
        self._channel.basic_publish(
            exchange=self.exchange,
            routing_key="",
            body=json.dumps(message, ensure_ascii=False),
        )
        return message

    def info(self, content: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish("INFO", content, **kwargs)

    def error(self, content: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish("ERROR", content, **kwargs)

    def generated(self, content: str, file_type: str, file_path: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish("GENERATED", content, file_type=file_type, file_path=file_path, **kwargs)

    def close(self) -> None:
        if self._connection and not self._connection.is_closed:
            self._connection.close()
        self._connection = None
        self._channel = None

    def _connect(self) -> None:
        if pika is None or BlockingConnection is None or PlainCredentials is None:
            raise RabbitMQPublishError("pika is not installed. Run: pip install -r requirements.txt")

        if self._connection is None or self._connection.is_closed:
            credentials = PlainCredentials(self.user, self.password)
            self._connection = BlockingConnection(
                pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    connection_attempts=1,
                    socket_timeout=3,
                    blocked_connection_timeout=3,
                )
            )
            self._channel = self._connection.channel()
            self._channel.exchange_declare(exchange=self.exchange, exchange_type="fanout", durable=True)

    def _next_event_id(self) -> int:
        with self._lock:
            self._event_id += 1
            return self._event_id


class SafeWorkflowPublisher:
    """Best-effort publisher used by the Flask workflow."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.enabled = RABBITMQ_ENABLED
        self.required = RABBITMQ_REQUIRED
        self.last_error: str | None = None
        self._publisher = AgentPublisher(session_id) if self.enabled else None

    def info(self, content: str, **kwargs: Any) -> None:
        self._safe_publish("INFO", content, **kwargs)

    def error(self, content: str, **kwargs: Any) -> None:
        self._safe_publish("ERROR", content, **kwargs)

    def generated(self, content: str, file_type: str, file_path: str, **kwargs: Any) -> None:
        self._safe_publish("GENERATED", content, file_type=file_type, file_path=file_path, **kwargs)

    def publish_stage(self, stage: str, payload: dict[str, Any]) -> None:
        self.info(
            str(payload.get("message") or stage),
            stage=stage,
            **compact_payload(payload),
        )

    def close(self) -> None:
        if self._publisher:
            self._publisher.close()

    def _safe_publish(self, event_type: str, content: str, **kwargs: Any) -> None:
        if not self.enabled or self._publisher is None:
            return
        try:
            self._publisher.publish(event_type, content, **kwargs)
        except Exception as exc:
            self.last_error = str(exc)
            if self.required:
                raise RabbitMQPublishError(str(exc)) from exc
            self.enabled = False


def publish_generated_artifacts(
    publisher: SafeWorkflowPublisher,
    generator_path: Path,
    input_files: list[Path],
) -> None:
    publisher.generated("generator.py generated", "generator", str(generator_path.resolve()))
    for path in input_files:
        publisher.generated("input file generated", "input", str(path.resolve()))


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "message":
            continue
        if key in {"problem_info", "test_plan", "test_data_spec"}:
            compacted[key] = summarize(value)
        elif key in {"generator_path", "test_data_dir", "agent_trace_path"}:
            compacted[key] = str(value)
        elif key == "input_files":
            compacted[key] = [str(path) for path in value]
        else:
            compacted[key] = to_jsonable(value)
    return compacted


def summarize(value: Any, max_length: int = 1200) -> str:
    text = json.dumps(to_jsonable(value), ensure_ascii=False)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value
