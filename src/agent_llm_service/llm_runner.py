import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agent_llm_service.provider.llm.base import BaseLlmProvider, LlmResponse


class LLMRunner(BaseModel):
    provider: BaseLlmProvider
    fallback_models: list[str] = Field(
        default_factory=list,
        description="list['<slug>/<model_name>']; used if model fails or not provided",
    )  # list of models ['<slug>/<model_name>'];
    _mdl_idx: int = PrivateAttr(-1)  # internal index to track which fallback model to use next
    max_retries: int = 3
    retry_base_delay: float = 1.0

    model_config = {"arbitrary_types_allowed": True}

    def _classify_error(self, e: Exception) -> str:
        msg = str(e).lower()
        if any(k in msg for k in ("429", "rate limit", "too many requests", "quota")):
            return "rate_limit"
        if any(k in msg for k in ("timeout", "timed out")):
            return "timeout"
        if any(
            k in msg
            for k in (
                "401",
                "403",
                "400",
                "unauthorized",
                "invalid api key",
                "bad request",
            )
        ):
            return "non_recoverable"
        return "unknown"

    def _get_next_model(self):
        if not self.fallback_models:
            raise ValueError("No fallback models available.")
        self._mdl_idx += 1
        return self.fallback_models[self._mdl_idx % len(self.fallback_models)]

    async def _call_with_retries(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        model: str | None = None,
        # if null models will be taken from fallback_models deque
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> LlmResponse:
        last_error: Exception | None = None

        if not model and not self.fallback_models:
            raise ValueError("No model provided and fallback_models is empty.")
        current_model = None
        for attempt in range(self.max_retries):
            try:
                # if first attempt & model provided, use it; else fallback to deque models
                current_model = model if attempt == 0 and model else self._get_next_model()
                logger.debug(
                    f"[LLMRunner] Attempt {attempt + 1}/{self.max_retries} using model: {current_model}"
                )
                return await self.provider.acall(
                    model=current_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                if error_type == "non_recoverable":
                    # todo: later just allow models from another provider for unauthorized errors
                    logger.warning(
                        f"[LLMRunner] Non-recoverable error on model {current_model}, skipping retries: {e}"
                    )
                    raise
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[LLMRunner] Error on model {current_model}: {e}. Retrying with a different model..."
                    )
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))

        raise last_error  # type: ignore[misc]

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        model: str | None = None,
        timeout: int | None = None,
    ) -> LlmResponse:
        try:
            return await self._call_with_retries(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                model=model,
            )
        except Exception as e:
            logger.error(f"[LLMRunner] All attempts failed for {model}: {e}")
            raise

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
        model: str | None = None,
    ) -> LlmResponse:
        return asyncio.run(
            self.acall(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                model=model,
            )
        )
