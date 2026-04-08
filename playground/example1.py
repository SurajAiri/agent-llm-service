import asyncio

from dotenv import load_dotenv

from agent_llm_service import LlmProviderConfig, LlmRunner, RawLlmProvider

load_dotenv()


async def run_single_prompt():
    # 1. Define the connection parameters to any provider mimicking OpenAI schemas
    config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY",  # Uses os.environ.get("GROQ_API_KEY")
        base_url="https://api.groq.com/openai/v1",
        enabled=True,
    )

    # 2. Build the Provider handler for dispatching calls
    provider = RawLlmProvider(config=[config])

    # 3. Create the Execution engine (automatically handles backoffs for simple 429 errors)
    runner = LlmRunner(provider=provider)

    # 4. Trigger the run. Model parsing: `<slug>/<version-id-on-provider>`
    response = await runner.acall(
        model="groq/openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How far is the moon?"},
        ],
    )

    print(f"Content: {response.content}")
    print(f"Token Usage: {response.usage}")


if __name__ == "__main__":
    asyncio.run(run_single_prompt())
