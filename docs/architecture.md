# Architecture of the Agent LLM Service

`agent_llm_service` acts as a generic and reusable backbone to interface GenAI code with third-party LLM providers. By detaching the orchestrating logic of standard chat clients into distinct responsibilities (Pools, Runners, and Extensible Tools), the framework scales flawlessly for a multi-agent system.

Below is the overarching component diagram and module breakdown.

## Modular Component Overview

```text
src/agent_llm_service/
 ├── core/          (Execution lifecycles, strategies, rate-limit recoveries)
 ├── providers/     (Network communication, unified interfaces, REST API details)
 ├── schemas/       (Data transfer objects, Configs logic strictly separated from execution)
 └── tools/         (Extensible classes mapping Python logic -> OpenAPI tool endpoints)
```

### 1. The `core` Domain (Runners & Execution Pools)
The classes in this directory encapsulate **strategy over the network requests**.
- **`LlmRunner`**: Your basic execution worker. Attempts to fulfill LLM calls. If a model is rate-limited (`429`), `LlmRunner` applies exponential backoffs. It fails gracefully on unrecoverable logic errors (`401` Unauthorized, `400` Bad Request) explicitly via HTTP status interpretations.
- **`LlmExecutionPool`**: Your advanced load-balancer. Provide this with an array (e.g., `fallback_models`). When a model repeatedly defaults (exceeding `failure_threshold`), it's placed on a "cooldown" phase. Successive calls immediately try alternative models rather than failing the execution context of an Agent.

### 2. The `providers` Domain
The package adopts a generic parsing layer to normalize diverse provider APIs into a strict `LlmResponse` model.
- **`BaseLlmProvider` (ABC)**: Contains the base implementation. Requires an `acall`/`call` method and configuration dictionaries.
- **`RawLlmProvider`**: Specifically adheres to the standard OpenAI endpoints style. Since most vendors currently proxy through OpenAI-compatible structures (including Groq, LiteLLM, vLLM, DeepSeek APIs), `RawLlmProvider` takes the configuration map (`LlmProviderConfig`) and fires requests mapped neatly.

### 3. The `schemas` Domain
Enforced Pydantic Models for internal state consistency (not parsing raw models).
- **`LlmProviderConfig`**: Contains environment configurations (like referencing OS `.env` tokens, parsing `headers`, endpoints, and slug-model mapping).

### 4. The `tools` Domain
LLMs rely natively on tool executions to query system state.
- **`BaseTool`**: An abstract interface that handles `execute(...)` and serialization (via `to_openai_schema()`).
- **`ToolRegistry`**: Manages all tools defined via the `BaseTool` class. `ToolRegistry.dispatch(...)` maps LLM instructions matching `id` and `arguments` mapping robustly to async Python tasks and normalizes return payloads into a unified `ToolResult`.

## Data Flow for a Standard Pool Request

1. **User/Agent System** invokes: `pool.acall(messages=...)`
2. `LlmExecutionPool` queries state: _Are any priority fallback models running fine?_ (Bypasses those on cooldown).
3. Selected model delegates to `RawLlmProvider.acall(...)`.
4. The provider handles actual API construction using `LlmProviderConfig` API credentials.
5. On success: Returns `LlmResponse` parsing raw tokens (and reasoning tags like `<think>...</think>`). If failing via HTTP limits, Provider raises a normalized payload back up to `LlmExecutionPool`.
6. Core Pool calculates threshold metrics and tries the next best model instantly.
