"""Errors that stop a run or reject a request, with the reason in the message."""

from __future__ import annotations


class RunError(RuntimeError):
    """A gate failed or the results folder cannot be trusted; the message says what and how on."""


class UsageError(ValueError):
    """The request cannot be run as asked, for example an unknown variable file stem."""
