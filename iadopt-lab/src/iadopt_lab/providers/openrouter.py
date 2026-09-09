"""OpenRouter-specific endpoint selection without inherited credentials."""

from .base import OpenAICompatibleAdapter


def create_adapter(profile: dict, api_key: str, **options) -> OpenAICompatibleAdapter:
    """Construct an OpenRouter adapter from frozen deployment settings.

    Args: profile: Explicit base URL/timeout; api_key: private OpenRouter key; options: mock transport.
    Returns: Separate one-request adapter.
    Raises: LabError for invalid configuration; KeyError for missing required profile fields.
    Side Effects: Allocates a client without sending a request.
    """
    return OpenAICompatibleAdapter("openrouter", profile["base_url"], api_key, profile["timeout_seconds"], **options)
