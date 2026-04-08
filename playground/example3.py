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

    async def execute(self, location: str, unit: str = "celsius", **kwargs) -> ToolResult:
        # Dummy weather retrieval logic:
        # result = await my_http_weather_fetcher(location)
        current_temp = "72F" if unit == "fahrenheit" else "22C"

        return ToolResult.ok(output=f"The current temperature in {location} is {current_temp}.")
