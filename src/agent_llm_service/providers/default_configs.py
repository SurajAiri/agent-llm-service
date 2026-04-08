from agent_llm_service.schemas.config import LlmProviderConfig

default_configs: list[LlmProviderConfig] = [
    LlmProviderConfig(
        slug="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env_var="GROQ_API_KEY",
        enabled=True,
    ),
    LlmProviderConfig(
        slug="gemini",
        name="Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env_var="GEMINI_API_KEY",
        # enabled=True,
    ),
]
