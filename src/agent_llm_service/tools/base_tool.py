"""
agent_llm_service/tools/base_tool.py
Abstract base for all agent tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standard result returned by every tool."""

    success: bool
    output: str  # always a string for LLM consumption
    error: str | None = None
    metadata: dict = {}

    @classmethod
    def ok(cls, output: str, metadata: dict | None = None) -> "ToolResult":
        return cls(success=True, output=output, metadata=metadata or {})

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, output=f"ERROR: {error}", error=error)


class BaseTool(ABC):
    """Every tool implements name, description, parameter schema, and execute."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def _parameters_schema(self) -> dict: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...

    def to_openai_schema(self) -> dict:
        """Return an OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters_schema(),
            },
        }
