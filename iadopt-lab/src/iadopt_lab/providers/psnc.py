"""PSNC-specific endpoint selection; no service-level decomposition request."""

from .base import OpenAICompatibleAdapter


def create_adapter(profile: dict, api_key: str, *, base_url_override: str | None = None,
                   **options) -> OpenAICompatibleAdapter:
    """Construct an independent PSNC SDK adapter."""
    return OpenAICompatibleAdapter("psnc", base_url_override or profile["default_base_url"],
                                   api_key, profile["timeout_seconds"], **options)
