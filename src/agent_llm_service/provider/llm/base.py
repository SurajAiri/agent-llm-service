from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agent_llm_service.models.llm_provider_config import LlmProviderConfig


class ToolCallRequest(BaseModel):
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class LlmResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCallRequest] = []
    reasoning_content: str | None = None  # optional field for reasoning content
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"  # "stop" or "tool_calls_exhausted"

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class BaseLlmProvider(BaseModel):
    config: list[LlmProviderConfig]

    def _get_provider_config(self, slug: str) -> LlmProviderConfig:
        """Get provider configuration by slug."""
        for cfg in self.config:
            if cfg.slug == slug:
                return cfg
        raise ValueError(f"No provider config found for slug: {slug}")

    def _separate_slug_model(self, model: str) -> tuple[str, str]:
        """Separate slug and model name from a string in <slug>/<model_name> format."""
        parts = model.split("/")
        if len(parts) < 2:
            raise ValueError(
                f"Model name must be in <slug>/<model_name> format, got: {model}"
            )
        return parts[0], "/".join(parts[1:])

    def add_provider_config(self, provider_config: LlmProviderConfig):
        """Add a new provider configuration to the existing list."""
        self.config.append(provider_config)

    @property
    def provider_configs(self) -> list[LlmProviderConfig]:
        """Get the list of provider configurations."""
        return self.config

    @abstractmethod
    def get_available_models(self, slug: str):
        """Get available models from the provider."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> LlmResponse:
        """Call the LLM provider with the given model and return the response."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def acall(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> LlmResponse:
        """Asynchronously call the LLM provider with the given model and return the response."""
        raise NotImplementedError("Subclasses must implement this method.")
