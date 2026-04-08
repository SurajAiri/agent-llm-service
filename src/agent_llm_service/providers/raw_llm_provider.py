import asyncio
import json
import re

import httpx
from loguru import logger

from agent_llm_service.schemas.config import LlmProviderConfig

from .base import BaseLlmProvider, LlmResponse, ToolCallRequest


class RawLlmProvider(BaseLlmProvider):
    async def get_available_models(self, slug: str) -> dict:
        """Return a dictionary of available models."""
        config = self._get_provider_config(slug)
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {config.api_key}"}) as client:
            response = await client.get(f"{config.base_url}/models")
            response.raise_for_status()
            return response.json()

    async def list_models(self, slug: str) -> list[str]:
        """Return a list of available model names."""
        models_data = await self.get_available_models(slug)
        return [model["id"] for model in models_data.get("data", [])]

    async def acall(
        self,
        model: str,  # with provider slug <slug>/<model_name> format
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> LlmResponse:
        """Call the LLM provider with the given model and return the response."""
        try:
            slug, mdl = self._separate_slug_model(model)
            config = self._get_provider_config(slug=slug)
            payload = {
                "messages": messages,
                "model": mdl,
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            if max_tokens:
                payload["max_tokens"] = max_tokens

            headers = config.headers.copy()
            headers["Content-Type"] = "application/json"
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Raw response from LLM provider {model}: {json.dumps(data)}")
                return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error while calling LLM provider {model}: {e.response.status_code} - {e.response.text}"
            )
            raise RuntimeError(f"LLM provider returned HTTP error {e.response.status_code}") from e
        except Exception as e:
            logger.exception(f"Error while calling LLM provider {model}: {str(e)}")
            raise RuntimeError(f"Error calling LLM provider: {str(e)}") from e

    @staticmethod
    def _parse_response(data: dict) -> LlmResponse:
        """Convert a raw OpenAI-style JSON dict → ``LLMResponse``."""

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"LLM API returned no choices: {json.dumps(data)[:500]}")

        message = choices[0].get("message", {})

        # --- Text content ---
        content = message.get("content") or None
        reasoning = message.get("reasoning")
        # if reasoning is inside <think> tags in the content, extract it and remove from content
        if content and reasoning is None:
            if "<think>" in content and "</think>" in content:
                reasoning = content.split("<think>")[1].split("</think>")[0].strip()

                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or None

        # --- Tool calls ---
        tool_calls: list[ToolCallRequest] = []
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            for tc in raw_tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tc_name = func.get("name", "")

                # Arguments come as a JSON string from the API
                raw_args = func.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to parse tool-call arguments for {}: {}",
                        tc_name,
                        raw_args,
                    )
                    arguments = {}

                if not tc_name:
                    logger.warning("Skipping tool call with empty name: {}", tc)
                    continue

                tool_calls.append(
                    ToolCallRequest(
                        id=tc_id,
                        name=tc_name,
                        arguments=arguments,
                    )
                )

        # --- Usage ---
        usage: dict[str, int] = {}
        raw_usage = data.get("usage")
        if raw_usage and isinstance(raw_usage, dict):
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

        logger.debug(
            f"LlmApi response | content_len={len(content) if content else 0} tool_calls={len(tool_calls)} usage={usage}",  # noqa: E501
        )

        return LlmResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            reasoning_content=reasoning,
            finish_reason=choices[0].get("finish_reason", "stop"),
        )

    def call(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        timeout: int | None = None,
    ) -> LlmResponse:
        """Synchronous wrapper around the asynchronous call method."""
        return asyncio.run(
            self.acall(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        )
