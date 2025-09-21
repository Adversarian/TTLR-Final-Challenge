"""Logfire instrumentation helpers for the agent."""

from __future__ import annotations

import os

import logfire


_LOGFIRE_READY = False


def _configure_logfire() -> None:
    """Configure Logfire instrumentation if it has not been configured yet."""

    token = os.getenv("LOGFIRE_API_KEY")
    if token:
        logfire.configure(token=token)
    else:
        logfire.configure()


def _ensure_logfire() -> None:
    """Initialize Logfire once for the process."""

    global _LOGFIRE_READY
    if not _LOGFIRE_READY:
        _configure_logfire()
        _LOGFIRE_READY = True


__all__ = ["_ensure_logfire"]
