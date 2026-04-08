import os

from loguru import logger
from pydantic import BaseModel, Field

# note: here is is_custom & base_url that are useful
# if you are using litellm based provider (which is not implemented yet)
# but for custom platforms and RawLlmProvider, you have to specify the base_url


class LlmProviderConfig(BaseModel):
    name: str
    slug: str
    api_key_env_var: str  # this will be mapped in .env file for api key
    base_url: str
    headers: dict = Field(default_factory=dict)
    adapter: str = Field(default="openai")
    enabled: bool = Field(default=False)
    models: list[str] = Field(
        default_factory=list
    )  # must follow format `<slug>/<model_id>`, e.g. `groq/llm-1`

    @property
    def api_key(self) -> str:
        """Get the API key from environment variable."""
        if not self.enabled:
            raise ValueError(f"Provider {self.name} is not enabled.")

        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key for {self.name} not found."
                f" Please set the environment variable: {self.api_key_env_var}"
            )
        return api_key


if __name__ == "__main__":
    # Example usage
    config = LlmProviderConfig(
        name="Groq",
        slug="groq",
        api_key_env_var="GROQ_API_KEY",
        enabled=True,
        base_url="https://api.groq.com/openai/v1",
        models=["groq/llm-1", "groq/llm-2"],
    )
    print(config)
