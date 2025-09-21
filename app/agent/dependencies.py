"""Dependency definitions for the shopping assistant agent."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass
class AgentDependencies:
    """Runtime dependencies passed to the agent on every run."""

    session: AsyncSession
    session_factory: async_sessionmaker[AsyncSession]


__all__ = ["AgentDependencies"]
