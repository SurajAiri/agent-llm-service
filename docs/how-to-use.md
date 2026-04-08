# How To Use the Agent LLM Service

This library simplifies connecting multi-agent logic into LLMs by abstracting away the provider intricacies and enforcing rate limit resilience, structured executions, and easy-to-use Tool Calling wrappers.

---

## Example 1: Basic LLM Runner Setup

You can map one primary configuration block (`LlmProviderConfig`) representing an API environment (e.g., Groq, OpenAI, Databricks, vLLM).

```python
import asyncio
from agent_llm_service import LlmProviderConfig, RawLlmProvider, LlmRunner

async def run_single_prompt():
    # 1. Define the connection parameters to any provider mimicking OpenAI schemas
    config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY", # Uses os.environ.get("GROQ_API_KEY")
        base_url="https://api.groq.com/openai/v1",
        enabled=True
    )
    
    # 2. Build the Provider handler for dispatching calls
    provider = RawLlmProvider(config=[config])
    
    # 3. Create the Execution engine (automatically handles backoffs for simple 429 errors)
    runner = LlmRunner(provider=provider)

    # 4. Trigger the run. Model parsing: `<slug>/<version-id-on-provider>`
    response = await runner.acall(
        model="groq/llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "How far is the moon?"}]
    )
    
    print(f"Content: {response.content}")
    print(f"Token Usage: {response.usage}")

if __name__ == "__main__":
    asyncio.run(run_single_prompt())
```

---

## Example 2: Resilience via `LlmExecutionPool` (Failovers)

Building autonomous systems typically crashes immediately when an API rate limit is exceeded for one service. `LlmExecutionPool` resolves this by assigning a set of **fallback models**. If a preferred model fails consecutively (exceeding `failure_threshold`), it marks that model on cooldown and immediately switches to the next fallback seamlessly.

```python
import asyncio
from agent_llm_service import LlmProviderConfig, RawLlmProvider, LlmExecutionPool

async def run_pool():
    # Configure multiple providers
    groq_config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        enabled=True
    )
    gemini_config = LlmProviderConfig(
        name="Gemini Config",
        slug="gemini",
        api_key_env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        enabled=True
    )
    
    provider = RawLlmProvider(config=[groq_config, gemini_config])
    
    # Setup Pool with round-robin fallbacks
    pool = LlmExecutionPool(
        provider=provider,
        fallback_models=[
            "groq/llama3-70b-8192",   # Try first
            "groq/llama3-8b-8192",    # Fallback to faster, lower-tier groq
            "gemini/gemini-pro"       # Failover completely to Gemini
        ],
        failure_threshold=3,       # Switch models after 3 consecutive failures
        cooldown_duration=60.0     # Put dead model on cooldown for 60 seconds
    )

    try:
        # Note how `model` string isn't required here; the Pool iterates automatically 
        # starting sequentially over the fallbacks list!
        response = await pool.acall(
            messages=[{"role": "user", "content": "Write a critical report on API resilience."}]
        )
        print("Success! Executed via model pooling.")
        print(response.content)

    except RuntimeError as e:
        print("All fallback models are currently exhausted or experiencing network closures.")

asyncio.run(run_pool())
```

---

## Example 3: Enabling Advanced Tool Calling

When building autonomous agents, they frequently require mapping the `tools` arguments directly to Python methods via Pydantic mapping schemas. You define everything cleanly inheriting from `BaseTool`.

### 1. Define a tool:
```python
from agent_llm_service import BaseTool, ToolResult

class GetWeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_current_weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a specific location."

    def _parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }

    async def execute(self, location: str, unit: str = "celsius", **kwargs) -> ToolResult:
        # Dummy weather retrieval logic:
        # result = await my_http_weather_fetcher(location)
        current_temp = "72F" if unit == "fahrenheit" else "22C"
        
        return ToolResult.ok(
            output=f"The current temperature in {location} is {current_temp}."
        )
```

### 2. Dispatch via Registry seamlessly

```python
import asyncio
from agent_llm_service import ToolRegistry, LlmRunner, RawLlmProvider, LlmProviderConfig

async def agent_execution_loop():
    provider = RawLlmProvider(config=[LlmProviderConfig(
        name="Groq", slug="groq", api_key_env_var="GROQ_API_KEY", base_url="https://api.groq.com/openai/v1", enabled=True
    )])
    runner = LlmRunner(provider=provider)
    
    # 1. Register tool into pool
    registry = ToolRegistry([GetWeatherTool()])

    messages = [{"role": "user", "content": "What's the weather in Seattle right now?"}]

    # 2. Tell LLM what tools are available (serialize cleanly mapping configurations)
    response = await runner.acall(
        model="groq/llama3-8b-8192",
        messages=messages,
        tools=registry.to_openai_schemas()
    )

    # 3. Intercept and execute callback requests requested natively by LLM response
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            print(f"LLM wants to run: {tool_call.name} with {tool_call.arguments}")
            
            # Dispatch exactly requested parameters automatically against your Python codebase
            tool_result = await registry.dispatch(
                tool_name=tool_call.name, 
                arguments=tool_call.arguments
            )
            
            print(f"Tool executed. Response: {tool_result.output}")
            
            # You can append the function/tool result to messages dict and send the query back
            # to let the LLM analyze your system tool fetch.

if __name__ == "__main__":
    asyncio.run(agent_execution_loop())
```