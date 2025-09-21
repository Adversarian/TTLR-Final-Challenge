"""Dependency definitions for the shopping assistant agent."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class AgentDependencies:
    """Runtime dependencies passed to the agent on every run."""

    session: AsyncSession


__all__ = ["AgentDependencies"]
