"""One-request provider adapters; orchestration owns all retry decisions."""

from .base import OpenAICompatibleAdapter, build_request

__all__ = ["OpenAICompatibleAdapter", "build_request"]
