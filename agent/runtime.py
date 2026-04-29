from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    step: int
    phase: str
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    agent_name: str
    events: list[AgentEvent] = field(default_factory=list)

    def add(self, phase: str, status: str, message: str, **data: Any) -> None:
        self.events.append(
            AgentEvent(
                step=len(self.events) + 1,
                phase=phase,
                status=status,
                message=message,
                data=compact_data(data),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events": [asdict(event) for event in self.events],
        }

    def save(self, trace_dir: Path, filename_prefix: str = "trace") -> Path:
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = trace_dir / f"{filename_prefix}_{timestamp}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class AgentTool:
    name = "tool"

    def describe(self) -> str:
        return self.name


def compact_data(data: dict[str, Any], max_text_length: int = 2000) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_text_length:
            compacted[key] = value[:max_text_length] + "...[truncated]"
        elif isinstance(value, Path):
            compacted[key] = str(value)
        elif isinstance(value, list):
            compacted[key] = [str(item) if isinstance(item, Path) else item for item in value[:20]]
        else:
            compacted[key] = value
    return compacted
