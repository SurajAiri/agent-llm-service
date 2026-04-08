import asyncio

from dotenv import load_dotenv

from agent_llm_service import LlmExecutionPool, LlmProviderConfig, RawLlmProvider

load_dotenv()


async def run_pool():
    # Configure multiple providers
    groq_config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY",  # Uses os.environ.get("GROQ_API_KEY")
        base_url="https://api.groq.com/openai/v1",
        enabled=True,
    )
    gemini_config = LlmProviderConfig(
        name="Gemini Config",
        slug="gemini",
        api_key_env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        enabled=True,
    )

    provider = RawLlmProvider(config=[groq_config, gemini_config])

    # Setup Pool with round-robin fallbacks
    pool = LlmExecutionPool(
        provider=provider,
        fallback_models=[
            "groq/llama3-70b-8192",  # Try first
            # "groq/openai/gpt-oss-120b",  # Fallback to faster, lower-tier groq
            "gemini/gemini-3.1-flash-lite-preview",  # Failover completely to Gemini
        ],
        failure_threshold=3,  # Switch models after 3 consecutive failures
        cooldown_duration=60.0,  # Put dead model on cooldown for 60 seconds
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
