import asyncio
import random
from typing import Any

from loguru import logger
from pydantic import BaseModel

from agent_llm_service.providers.base import BaseLlmProvider, LlmResponse


class LlmRunner(BaseModel):
    provider: BaseLlmProvider

    # ── Error classification ──────────────────────────────────────────────────

    def _classify_error(self, e: Exception) -> str:
        msg = str(e).lower()
        if any(k in msg for k in ("401", "unauthorized", "invalid api key")):
            return "unauthorized"
        if any(k in msg for k in ("403", "400", "bad request")):
            return "non_recoverable"
        if any(k in msg for k in ("429", "rate limit", "too many requests", "quota")):
            return "rate_limit"
        if any(k in msg for k in ("timeout", "timed out")):
            return "timeout"
        return "unknown"

    def _get_retry_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay with jitter."""
        return base_delay * (2**attempt) + random.uniform(0, 1)

    async def acall(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> LlmResponse:
        for attempt in range(max_retries):
            try:
                return await self.provider.acall(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except Exception as e:
                error_type = self._classify_error(e)
                logger.error(f"Error calling model {model} ({error_type}): {e}")
                if error_type in ("unauthorized", "non_recoverable"):
                    raise
                if error_type in ("rate_limit", "timeout"):
                    delay = self._get_retry_delay(attempt, base_delay)
                    logger.warning(f"Retrying after {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

        raise RuntimeError(f"Failed to call model {model} after {max_retries} attempts.")

    def call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> LlmResponse:
        """Synchronous wrapper around acall."""
        return asyncio.run(
            self.acall(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        )
