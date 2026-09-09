"""PSNC-specific endpoint selection; no service-level decomposition request."""

from .base import OpenAICompatibleAdapter


def create_adapter(profile: dict, api_key: str, *, base_url_override: str | None = None,
                   **options) -> OpenAICompatibleAdapter:
    """Construct an independent PSNC SDK adapter.

    Args: profile: Frozen default URL/timeout; api_key: PSNC key; base_url_override: explicit optional URL;
        options: offline transport injection.
    Returns: Separate one-request adapter with /v1 included exactly once.
    Raises: LabError for invalid values; KeyError for incomplete profile.
    Side Effects: Allocates a client without network or secret persistence.
    """
    return OpenAICompatibleAdapter("psnc", base_url_override or profile["default_base_url"],
                                   api_key, profile["timeout_seconds"], **options)
