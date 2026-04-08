import asyncio
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agent_llm_service.provider.llm.base import BaseLlmProvider, LlmResponse


class LlmExecutionPool(BaseModel):
    provider: BaseLlmProvider
    fallback_models: list[str] = Field(default_factory=list)
    failure_threshold: int = 3  # consecutive failures before cooldown
    cooldown_duration: float = 60.0  # seconds

    model_config = {"arbitrary_types_allowed": True}

    _mdl_idx: int = PrivateAttr(default=-1)
    _model_failures: dict[str, int] = PrivateAttr(default_factory=dict)
    _cooldown_models: dict[str, float] = PrivateAttr(
        default_factory=dict
    )  # model -> expires_at, inf = permanent

    # ── Cooldown management ───────────────────────────────────────────────────

    def _is_on_cooldown(self, model: str) -> bool:
        expires_at = self._cooldown_models.get(model)
        if expires_at is None:
            return False
        if expires_at == float("inf"):
            return True
        if time.monotonic() < expires_at:
            return True
        # cooldown expired, clean up
        del self._cooldown_models[model]
        self._model_failures.pop(model, None)
        return False

    def _put_on_cooldown(self, model: str, duration: float) -> None:
        expires_at = float("inf") if duration == float("inf") else time.monotonic() + duration
        self._cooldown_models[model] = expires_at
        label = "permanent" if duration == float("inf") else f"{duration:.0f}s"
        logger.warning(f"[LLMRunner] Model {model} put on cooldown ({label})")

    def _put_provider_on_cooldown(self, slug: str) -> None:
        """Put all models from a provider slug on permanent cooldown."""
        for model in self.fallback_models:
            if model.split("/")[0] == slug:
                self._put_on_cooldown(model, float("inf"))
        logger.error(
            f"[LLMRunner] Provider '{slug}' unauthorized — all its models on permanent cooldown. "
            f"Replace the API key and restart."
        )

    def _record_failure(self, model: str) -> None:
        self._model_failures[model] = self._model_failures.get(model, 0) + 1
        if self._model_failures[model] >= self.failure_threshold:
            self._put_on_cooldown(model, self.cooldown_duration)

    def _record_success(self, model: str) -> None:
        self._model_failures.pop(model, None)

    # ── Model selection ───────────────────────────────────────────────────────

    def _get_next_model(self) -> str:
        if not self.fallback_models:
            raise ValueError("No fallback models configured.")

        # try every model once to find one not on cooldown
        for _ in range(len(self.fallback_models)):
            self._mdl_idx += 1
            model = self.fallback_models[self._mdl_idx % len(self.fallback_models)]
            if not self._is_on_cooldown(model):
                return model

        raise RuntimeError("All fallback models are on cooldown.")

    def _get_model_order(self, preferred: str | None = None) -> list[str]:
        models = self.fallback_models

        if not models:
            return []

        # global rotation
        self._mdl_idx += 1
        start = self._mdl_idx % len(models)

        rotated = models[start:] + models[:start]

        if preferred:
            return [preferred] + [m for m in rotated if m != preferred and not self._is_on_cooldown(m)]
        return [m for m in rotated if not self._is_on_cooldown(m)]

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

    # ── Core call logic ───────────────────────────────────────────────────────

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        model: str | None = None,
        timeout: int | None = None,
    ) -> LlmResponse:
        if not model and not self.fallback_models:
            raise ValueError("No model provided and fallback_models is empty.")

        models_to_try = self._get_model_order(model)

        last_error = None

        for m in models_to_try:
            try:
                logger.debug(f"[LLMRunner] Trying model: {m}")

                response = await self.provider.acall(
                    model=m,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )

                self._record_success(m)
                return response

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

                if error_type == "unauthorized":
                    slug = m.split("/")[0]
                    self._put_provider_on_cooldown(slug)
                    raise

                if error_type == "non_recoverable":
                    raise

                # just mark failure and move on immediately
                self._record_failure(m)
                logger.warning(f"[LLMRunner] {m} failed ({error_type}), switching...")

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def call(self, messages: list[dict[str, Any]], **kwargs) -> LlmResponse:
        return asyncio.run(self.acall(messages, **kwargs))
