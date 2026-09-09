"""Strict lexical shape and objective cross-reference validation."""

from .lexical import (
    ValidationResult,
    canonicalize_prediction,
    empty_prediction,
    load_schema_bytes,
    readable_prediction,
    system_display_label,
    validate_candidate,
    validate_prediction,
)

__all__ = ["ValidationResult", "canonicalize_prediction", "empty_prediction",
           "load_schema_bytes", "readable_prediction", "system_display_label", "validate_candidate",
           "validate_prediction"]
