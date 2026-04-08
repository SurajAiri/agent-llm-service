# [Agent LLM Service](http://github.com/SurajAiri/agent-llm-service)

**Agent LLM Service** is a robust, clean, and extensible Python library designed for executing LLM requests across multiple language models (e.g., OpenAI, Groq, Gemini). It offers reliable fallback mechanics, model round-robins, API rate-limit management, built-in tool/function-calling structures, and exponential backoff retry logic.

It is built as the core execution engine for any multi-agent system, strictly adhering to modern Python typing (`Pydantic` v2) and asynchronous request handling (`httpx`, `asyncio`).

## Key Features

- **Execution Pool & Failover Strategy:** Automatically cycle through fallback models when encountering rate limits or temporary provider issues (`LlmExecutionPool`).
- **Resilient Polling & Backoffs:** Built-in retry with exponential backoff (`LlmRunner`).
- **Modular Provider Interface:** Easily write adapters and unified `RawLlmProvider` handlers that work with OpenAI-spec endpoints.
- **Native Tool Calling Definitions:** Extensively typed schemas to define custom tools (`BaseTool`) and dispatch LLM decisions automatically mappings to real functions (`ToolRegistry`).
- **Asynchronous by Default:** Built utilizing async (`httpx.AsyncClient`) for scaling safely across multi-agent fleets.

## Getting Started

### Installation
Ensure you are using `Python >= 3.12`. Install dependencies using [uv](https://github.com/astral-sh/uv) or pip:
```bash
uv add agent-llm-service 

# or you can also use with pip
# pip install agent-llm-service
```

### Quick Usage

```python
from agent_llm_service import LlmProviderConfig, RawLlmProvider, LlmRunner
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


async def main():
    # Configure the provider (e.g., Groq via OpenAI schema)
    config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        enabled=True,
    )

    provider = RawLlmProvider(config=[config])
    runner = LlmRunner(provider=provider)

    # Async LLM Run
    response = await runner.acall(
        model="groq/openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(response.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

## Documentation
_**Note:** If not shown directly on PyPi package, try seeing in [Github repo: SurajAiri/Agent-LLM-Service](http://github.com/SurajAiri/agent-llm-service)_
- **[Architecture Review](https://github.com/surajairi/agent-llm-service/docs/architecture.md):** Deep dive into the modular structure, object domains, and the pool failover strategies.
- **[How to Use (Examples & Tool Calling)](https://github.com/surajairi/agent-llm-service/docs/how-to-use.md):** Extensive recipes on handling single LLM runs, pools, and tool registry integrations.

## Contributing
1. Fork the repository and create an issue.
2. Ensure you run the linter (`uv run ruff check src --fix`).
3. Submit a PR.

## License
MIT License.
