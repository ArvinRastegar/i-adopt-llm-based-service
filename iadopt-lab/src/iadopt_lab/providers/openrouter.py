"""OpenRouter-specific endpoint selection without inherited credentials."""

from .base import OpenAICompatibleAdapter


def create_adapter(profile: dict, api_key: str, *, base_url_override: str | None = None,
                   **options) -> OpenAICompatibleAdapter:
    """Construct an OpenRouter adapter from frozen deployment settings."""
    return OpenAICompatibleAdapter("openrouter", base_url_override or profile["base_url"],
                                   api_key, profile["timeout_seconds"], **options)
