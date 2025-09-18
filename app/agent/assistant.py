from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import List, Optional

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

from app.config import settings
from app.db import get_pool
from app.models.chat import ChatRequest, ChatResponse

LOGFIRE_AGENT_INSTRUMENTED = False


@dataclass
class AgentDependencies:
    """Runtime dependencies available to agent tools."""

    database_url: Optional[str] = None

    @property
    def pool(self):
        return get_pool()


class ProductLookupArgs(BaseModel):
    product_name: Optional[str] = None
    base_random_key: Optional[str] = None
    member_random_key: Optional[str] = None
    limit: int = 10


class ProductMatch(BaseModel):
    random_key: str
    persian_name: Optional[str] = None
    english_name: Optional[str] = None
    matched_via: str


@lru_cache(maxsize=1)
def get_agent() -> Agent[AgentDependencies, ChatResponse]:
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
    else:
        os.environ.pop("OPENAI_BASE_URL", None)

    effort = settings.openai_reasoning_effort
    if effort and effort not in {"minimal", "low", "medium", "high"}:
        effort = None

    model_settings = OpenAIChatModelSettings(
        openai_reasoning_effort=effort,
    )

    model = OpenAIChatModel(
        settings.openai_chat_model,
        settings=model_settings,
    )

    agent = Agent(
        model=model,
        instructions=[
            "You are a focused Torob shopping assistant.",
            "Always answer with the JSON object {message, base_random_keys, member_random_keys} and nothing else.",
            "For the sanity check 'ping', respond with message 'pong' and both key lists null (None).",
            "When the user says 'return base random key: <VALUE>', immediately return base_random_keys=[<VALUE>], message null, member_random_keys null without validation.",
            "When the user says 'return member random key: <VALUE>', immediately return member_random_keys=[<VALUE>], message null, base_random_keys null without validation.",
            "Call tools to consult the database before recommending keys.",
            "Populate lookup_products arguments with precise keywords or identifiers extracted from the conversation (avoid polite filler).",
            "Prefer results that match explicit base/member keys or strong semantic similarity. Keep message null unless no result is found.",
            "Be brief: one sentence in message when needed; otherwise return null.",
            "Never guess—only emit keys that the tools confirm.",
        ],
        output_type=ChatResponse,
        deps_type=AgentDependencies,
    )

    global LOGFIRE_AGENT_INSTRUMENTED
    if not LOGFIRE_AGENT_INSTRUMENTED:
        try:
            logfire.instrument_pydantic_ai(agent)
        except Exception as exc:  # pragma: no cover - best effort
            logfire.warning("instrument_pydantic_ai_failed", error=str(exc))
        LOGFIRE_AGENT_INSTRUMENTED = True

    @agent.tool
    def lookup_products(
        ctx: RunContext[AgentDependencies], args: ProductLookupArgs
    ) -> List[ProductMatch]:
        """Lookup base products given structured filters extracted by the agent."""
        args.limit = max(1, min(args.limit, 25))
        pool = ctx.deps.pool

        base_key = args.base_random_key.strip() if args.base_random_key else None
        member_key = args.member_random_key.strip() if args.member_random_key else None
        search_text = args.product_name.strip() if args.product_name else None

        filters: List[str] = []
        params: List[str] = []

        if base_key:
            filters.append("bp.random_key = %s")
            params.append(base_key)
        if member_key:
            filters.append("m.random_key = %s")
            params.append(member_key)
        if search_text:
            filters.append(
                "(bp.persian_name ILIKE %s OR bp.english_name ILIKE %s)"
            )
            like = f"%{search_text}%"
            params.extend([like, like])

        if not filters:
            return []

        where_clause = f"WHERE {' AND '.join(filters)}"
        base_query = f"""
            SELECT DISTINCT bp.random_key,
                            bp.persian_name,
                            bp.english_name,
                            m.random_key AS member_random_key
            FROM base_products bp
            LEFT JOIN members m ON m.base_random_key = bp.random_key
            {where_clause}
            ORDER BY bp.random_key
            LIMIT %s
        """
        params_with_limit = params + [args.limit]

        results: List[ProductMatch] = []
        seen: set[str] = set()

        def add_rows(rows, default_tag: str) -> None:
            for record in rows:
                random_key, persian, english, member_match, *_ = record
                if random_key in seen:
                    continue
                matched_via = default_tag
                if base_key and random_key == base_key:
                    matched_via = "base_random_key"
                elif member_key and member_match and member_match == member_key:
                    matched_via = "member_random_key"
                results.append(
                    ProductMatch(
                        random_key=random_key,
                        persian_name=persian,
                        english_name=english,
                        matched_via=matched_via,
                    )
                )
                seen.add(random_key)

        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(base_query, params_with_limit)
                add_rows(cursor.fetchall(), "name_match" if search_text else "candidate")

                if search_text and len(results) < args.limit:
                    fts_query = """
                        SELECT DISTINCT bp.random_key,
                                        bp.persian_name,
                                        bp.english_name,
                                        m.random_key AS member_random_key,
                                        ts_rank_cd(
                                            to_tsvector('simple', coalesce(bp.persian_name,'') || ' ' || coalesce(bp.english_name,'')),
                                            plainto_tsquery('simple', %s)
                                        ) AS rank
                        FROM base_products bp
                        LEFT JOIN members m ON m.base_random_key = bp.random_key
                        WHERE to_tsvector('simple', coalesce(bp.persian_name,'') || ' ' || coalesce(bp.english_name,'')) @@ plainto_tsquery('simple', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                    """
                    cursor.execute(fts_query, [search_text, search_text, args.limit])
                    add_rows(cursor.fetchall(), "fts")

                if search_text and len(results) < args.limit:
                    trigram_query = """
                        SELECT DISTINCT bp.random_key,
                                        bp.persian_name,
                                        bp.english_name,
                                        m.random_key AS member_random_key,
                                        GREATEST(
                                            similarity(coalesce(bp.persian_name,''), %s),
                                            similarity(coalesce(bp.english_name,''), %s)
                                        ) AS score
                        FROM base_products bp
                        LEFT JOIN members m ON m.base_random_key = bp.random_key
                        WHERE GREATEST(
                            similarity(coalesce(bp.persian_name,''), %s),
                            similarity(coalesce(bp.english_name,''), %s)
                        ) > 0.1
                        ORDER BY score DESC
                        LIMIT %s
                    """
                    cursor.execute(
                        trigram_query,
                        [search_text, search_text, search_text, search_text, args.limit],
                    )
                    add_rows(cursor.fetchall(), "fuzzy")

                embedding_literal: Optional[str] = None
                if (
                    search_text
                    and settings.openai_embed_model
                    and len(results) < args.limit
                ):
                    try:
                        client = _get_embedding_client()
                        embedding = client.embeddings.create(
                            model=settings.openai_embed_model,
                            input=search_text,
                        )
                        embedding_literal = _vector_literal(
                            embedding.data[0].embedding
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logfire.warning("embedding_query_failed", error=str(exc))

                if embedding_literal and len(results) < args.limit:
                    vector_query = """
                        SELECT DISTINCT bp.random_key,
                                        bp.persian_name,
                                        bp.english_name,
                                        m.random_key AS member_random_key,
                                        1 - (pe.embedding <=> %s::vector) AS similarity
                        FROM product_embeddings pe
                        JOIN base_products bp ON bp.random_key = pe.random_key
                        LEFT JOIN members m ON m.base_random_key = bp.random_key
                        ORDER BY similarity DESC
                        LIMIT %s
                    """
                    cursor.execute(vector_query, [embedding_literal, args.limit])
                    add_rows(cursor.fetchall(), "semantic")

        priority = {
            "base_random_key": 0,
            "member_random_key": 1,
            "semantic": 2,
            "fts": 3,
            "fuzzy": 4,
            "name_match": 5,
            "candidate": 6,
        }
        results.sort(key=lambda r: priority.get(r.matched_via, 7))
        return results

    return agent


async def run_chat(request: ChatRequest) -> ChatResponse:
    agent = get_agent()
    deps = AgentDependencies(database_url=settings.database_url)

    if not request.messages:
        return ChatResponse()

    conversation_lines = []
    for message in request.messages[:-1]:
        conversation_lines.append(f"{message.type}: {message.content}")

    latest_message = request.messages[-1]
    if conversation_lines:
        user_prompt = (
            "Conversation so far:\n"
            + "\n".join(conversation_lines)
            + "\n\nLatest user message:\n"
            + latest_message.content
        )
    else:
        user_prompt = latest_message.content

    result = await agent.run(user_prompt=user_prompt, deps=deps)
    return result.output


try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


_embedding_client = None


def _get_embedding_client():
    global _embedding_client
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    if _embedding_client is None:
        client_kwargs = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        _embedding_client = OpenAI(**client_kwargs)
    return _embedding_client


def _vector_literal(vector: List[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
