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
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        location = kwargs.get("location")
        unit = kwargs.get("unit", "celsius")

        # Dummy weather retrieval logic:
        # result = await my_http_weather_fetcher(location)
        current_temp = "72F" if unit == "fahrenheit" else "22C"

        return ToolResult.ok(output=f"The current temperature in {location} is {current_temp}.")


# ------------------------------------------------------------------------------
# calling logic (example usage of above tool in an agent loop)
# ------------------------------------------------------------------------------

import asyncio
from agent_llm_service import ToolRegistry, LlmRunner, RawLlmProvider, LlmProviderConfig

from dotenv import load_dotenv

load_dotenv()


async def agent_execution_loop():
    provider = RawLlmProvider(
        config=[
            LlmProviderConfig(
                name="Groq",
                slug="groq",
                api_key_env_var="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                enabled=True,
            )
        ]
    )
    runner = LlmRunner(provider=provider)

    # 1. Register tool into pool
    registry = ToolRegistry([GetWeatherTool()])

    messages = [{"role": "user", "content": "What's the weather in Seattle right now?"}]

    # 2. Tell LLM what tools are available (serialize cleanly mapping configurations)
    response = await runner.acall(
        model="groq/openai/gpt-oss-120b", messages=messages, tools=registry.to_openai_schemas()
    )

    # 3. Intercept and execute callback requests requested natively by LLM response
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            print(f"LLM wants to run: {tool_call.name} with {tool_call.arguments}")

            # Dispatch exactly requested parameters automatically against your Python codebase
            tool_result = await registry.dispatch(tool_name=tool_call.name, arguments=tool_call.arguments)

            print(f"Tool executed. Response: {tool_result.output}")

            # You can append the function/tool result to messages dict and send the query back
            # to let the LLM analyze your system tool fetch.


if __name__ == "__main__":
    asyncio.run(agent_execution_loop())
