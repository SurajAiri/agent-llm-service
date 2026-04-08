"""
core/tools/registry.py
Tool registry — maps tool names to instances, dispatches calls.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from agent_llm_service.tools.base_tool import BaseTool, ToolResult


class ToolRegistry:
    """
    Holds a set of tools and dispatches LLM tool-call requests to them.
    Workers receive a subset based on allowed_tools list.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Tool registered: {}", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def subset(self, allowed: list[str]) -> "ToolRegistry":
        """Return a new registry containing only the allowed tools."""
        if not allowed:
            return self  # empty means all tools
        filtered = [t for n, t in self._tools.items() if n in allowed]
        return ToolRegistry(filtered)

    def to_openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(tool_name)
        if tool is None:
            err = f"Unknown tool '{tool_name}'. Available: {self.names()}"
            logger.warning(err)
            return ToolResult.fail(err)
        try:
            logger.debug("Dispatching tool '{}' | args={}", tool_name, arguments)
            result = await tool.execute(**arguments)
            if result.success:
                logger.debug(
                    "Tool '{}' succeeded | output_len={}", tool_name, len(result.output)
                )
            else:
                logger.warning("Tool '{}' failed | error={}", tool_name, result.error)
            return result
        except Exception as exc:
            logger.exception("Tool '{}' raised exception: {}", tool_name, exc)
            return ToolResult.fail(str(exc))
