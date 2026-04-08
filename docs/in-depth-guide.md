# In-Depth Components Guide

This guide provides an in-depth look at the core components of the Agent LLM Service. It covers configuration, the provider implementation, and the execution engines that handle retries, fallbacks, and multi-model routing.

---

## 1. `LlmProviderConfig`

The `LlmProviderConfig` class (found in `src/agent_llm_service/schemas/config.py`) is a Pydantic model used to define the connection parameters for your LLM providers. Since the service proxies API requests in an OpenAI-compatible format, this configuration securely handles how to communicate with endpoints like Groq, LiteLLM, vLLM, or natively OpenAI.

### Key Fields:
- **`name`** (`str`): A human-readable name for the provider (e.g., `"Groq"`).
- **`slug`** (`str`): A unique identifier used to route model calls (e.g., `"groq"`). Model names passed to the runner should follow the `<slug>/<model_id>` format (e.g., `groq/llama3-8b-8192`).
- **`api_key_env_var`** (`str`): The name of the environment variable where the API key is stored (e.g., `"GROQ_API_KEY"`). The config dynamically resolves this at runtime via the `api_key` property so you don't hardcode secrets.
- **`base_url`** (`str`): The base endpoint of the provider's API (e.g., `"https://api.groq.com/openai/v1"`).
- **`headers`** (`dict`): Any custom HTTP headers required by the provider (defaults to an empty dict).
- **`adapter`** (`str`): The format adapter to use. (Defaults to `"openai"`).
- **`enabled`** (`bool`): Toggle switch for the provider. If `False`, attempting to retrieve the `api_key` will raise a `ValueError`.
- **`models`** (`list[str]`): A list of supported models in the `<slug>/<model_id>` format.

### Example:
```python
from agent_llm_service import LlmProviderConfig

config = LlmProviderConfig(
    name="Groq",
    slug="groq",
    api_key_env_var="GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1",
    enabled=True
)
```

---

## 2. `RawLlmProvider`

The `RawLlmProvider` (found in `src/agent_llm_service/providers/raw_llm_provider.py`) is the workhorse for network communications. It implements the `BaseLlmProvider` interface and is specifically designed to handle OpenAI-compatible REST APIs.

### Responsibilities:
- **Request Formatting**: Takes the uniform inputs (messages, tools, tokens, temperature) and translates them into the JSON payload expected by the `/chat/completions` endpoint.
- **Dynamic Routing**: Parses the incoming `model` string (e.g., `"groq/llama3-8b-8192"`) to extract the `slug` (`"groq"`) and the actual target model (`"llama3-8b-8192"`), retrieving the correct `LlmProviderConfig` dynamically.
- **Response Parsing (`_parse_response`)**: Normalizes the raw JSON response back into a strictly typed `LlmResponse` object.
  - **Reasoning Handling (`<think>` tags)**: If the LLM generates reasoning steps enclosed in `<think>...</think>` tags (common in deep-thinking models like DeepSeek), the provider parses this out into the `reasoning_content` field and strips it from the main `content` to keep the standard output clean.
  - **Tool Calls**: Safely unpacks stringified JSON arguments into Python dictionaries for any function calls requested by the LLM.
- **Discovery**: Provides utility methods like `get_available_models(slug)` and `list_models(slug)` that hit the `GET /models` endpoint of the provider to list active deployments.

### Methods:
- `acall(...)`: The asynchronous execution method using `httpx.AsyncClient`.
- `call(...)`: A synchronous wrapper around `acall`.

---

## 3. `LlmRunner`

The `LlmRunner` (found in `src/agent_llm_service/core/llm_runner.py`) is a straightforward execution engine that sits on top of a provider. Its primary purpose is to add **resilience and retry logic** to single-model workflows.

### How it works:
Instead of crashing when a provider hiccups, `LlmRunner` traps HTTP exceptions and analyzes them using its `_classify_error` method:
- **`unauthorized` (e.g., 401)** or **`non_recoverable` (e.g., 400, 403)**: The runner immediately aborts, as retrying will clearly not fix bad credentials or malformed payloads.
- **`rate_limit` (e.g., 429)** or **`timeout`**: The runner applies an **exponential backoff with jitter** (`_get_retry_delay`). It will pause and try again up to `max_retries` times.

### Configuration inside `acall`/`call`:
- `max_retries` (default `3`): How many times to retry rate limits.
- `base_delay` (default `1.0` seconds): The starting wait time before exponential scaling.

### When to use:
Use `LlmRunner` when you have a specific, reliable model in mind for a task and just need safety against sporadic network timeouts or rate limits.

---

## 4. `LlmExecutionPool`

The `LlmExecutionPool` (found in `src/agent_llm_service/core/llm_execution_pool.py`) is the advanced multi-model router designed for high-availability agentic systems. If one provider goes down or strictly rate-limits your organization, the Pool automatically routes to the next best alternative.

### Core Features:
- **`fallback_models`**: A prioritized list of models to use (e.g., `["groq/llm1", "groq/llm2", "gemini/llm3"]`).
- **Cooldown Management**:
  - The pool tracks sequential failures per model based on the `failure_threshold` (default `3`).
  - If a model fails 3 times in a row, it's put on a time-out for `cooldown_duration` (default `60.0` seconds), and traffic is instantly shifted to the next fallback model.
- **Permanent Blacklisting**: If a provider returns a `401 Unauthorized` or isn't properly configured (`not_found`), the Pool permanently blacklists (infinite cooldown) **all** models matching that provider's `slug` so it doesn't waste time trying alternative models on a dead API Key.
- **Auto-Rotation (`_get_model_order`)**: If you don't explicitly pass a `model` string when calling the pool, it will start with the first healthy model in the `fallback_models` array and seamlessly rotate traffic. Alternatively, you can pass a `preferred` model to attempt it first.

### Why it's powerful:
For long-running autonomous tasks, APIs inevitably fail. By combining the unified interface of `RawLlmProvider`, the standard retry mechanism, and the tier-based routing of `LlmExecutionPool`, your agent logic remains oblivious to infrastructure turbulence. It just receives the final `LlmResponse` from whichever model was healthy enough to answer.