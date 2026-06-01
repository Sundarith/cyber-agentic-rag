from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str = ""
    type: str = ""
    identifier: str = ""
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Chunk":
        identifier = str(row.get("identifier") or row.get("id") or "").upper()
        chunk_id = str(row.get("id") or identifier)
        known = {"id", "text", "source", "type", "identifier", "name"}
        return cls(
            id=chunk_id,
            text=str(row.get("text") or ""),
            source=str(row.get("source") or ""),
            type=str(row.get("type") or ""),
            identifier=identifier,
            name=str(row.get("name") or ""),
            metadata={k: v for k, v in row.items() if k not in known},
        )


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float
    tool: str
    reason: str = ""
    snippet: str = ""

    @property
    def identifier(self) -> str:
        return self.chunk.identifier or self.chunk.id


class ActionType(str, Enum):
    SEARCH = "search"
    OPEN = "open"
    EXPAND = "expand"
    FIND = "find"


@dataclass(frozen=True)
class Action:
    kind: ActionType
    value: str
    rationale: str
    limit: int | None = None


@dataclass
class AgentState:
    question: str
    actions: list[Action] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    opened_ids: set[str] = field(default_factory=set)
    step_logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Verification:
    supported: bool
    confidence: float
    missing: list[str]
    cited_ids: list[str]
