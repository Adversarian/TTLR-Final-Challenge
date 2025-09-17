from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import os

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

from app.config import settings
from app.db import get_pool
from app.models.chat import ChatRequest, ChatResponse


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
            "When calling lookup_products extract the meaningful keywords or product codes (e.g. D14) instead of the full sentence.",
            "Prefer results that match provided base keys, member keys, or explicit codes. Return the strongest match and keep message null unless no result is found.",
            "Be brief: one sentence in message when needed; otherwise return null.",
            "Never guess—only emit keys that the tools confirm.",
        ],
        output_type=ChatResponse,
        deps_type=AgentDependencies,
    )

    @agent.tool
    def lookup_products(
        ctx: RunContext[AgentDependencies], args: ProductLookupArgs
    ) -> List[ProductMatch]:
        """Lookup base products by base/member key or name fragment; returns unique base_random_keys."""
        args.limit = max(1, min(args.limit, 25))
        pool = ctx.deps.pool
        conditions: List[str] = []
        params: List[str] = []

        base_key = args.base_random_key.strip() if args.base_random_key else None
        member_key = args.member_random_key.strip() if args.member_random_key else None
        code_terms: List[str] = []
        name_terms: List[str] = []

        if args.product_name:
            text = args.product_name.strip()
            code_terms = list(
                {
                    match.upper()
                    for match in re.findall(
                        r"(?:کد|code)[:\-\s]*([A-Za-z0-9\-]{2,})",
                        text,
                        flags=re.IGNORECASE,
                    )
                }
            )
            tokens = re.findall(r"[\wآ-ی]+", text)
            stop_words = {
                "لطفا",
                "لطفاً",
                "لطفا،",
                "لطفاً،",
                "برای",
                "من",
                "را",
                "میخواهم",
                "می‌خواهم",
                "خواهید",
                "می",
                "یک",
                "ا",
                "را",
                "تهیه",
                "کنید",
                "بفرمایید",
                "خواهشا",
                "خواهشاً",
            }
            for token in tokens:
                if len(token) <= 2:
                    continue
                normalized = token.lower()
                if normalized in stop_words:
                    continue
                name_terms.append(token)

        if base_key:
            conditions.append("bp.random_key = %s")
            params.append(base_key)
        if member_key:
            conditions.append("m.random_key = %s")
            params.append(member_key)

        if code_terms:
            code_clause_parts: List[str] = []
            for code in code_terms:
                like = f"%{code}%"
                code_clause_parts.append(
                    "bp.persian_name ILIKE %s OR bp.english_name ILIKE %s"
                )
                params.extend([like, like])
            conditions.append("(" + " OR ".join(code_clause_parts) + ")")

        if name_terms:
            name_clause_parts: List[str] = []
            for term in name_terms[:4]:  # cap to avoid huge OR clause
                like = f"%{term}%"
                name_clause_parts.append(
                    "bp.persian_name ILIKE %s OR bp.english_name ILIKE %s"
                )
                params.extend([like, like])
            conditions.append("(" + " OR ".join(name_clause_parts) + ")")

        if not conditions:
            return []

        where_clause = f"WHERE {' AND '.join(conditions)}"
        query = f"""
            SELECT DISTINCT bp.random_key, bp.persian_name, bp.english_name
            FROM base_products bp
            LEFT JOIN members m ON m.base_random_key = bp.random_key
            {where_clause}
            ORDER BY bp.random_key
            LIMIT %s
        """
        params_with_limit = params + [args.limit]

        results: List[ProductMatch] = []
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params_with_limit)
                for record in cursor.fetchall():
                    matched_via = "candidate"
                    random_key = record[0]
                    full_name = (record[1] or "") + " " + (record[2] or "")
                    if base_key and random_key == base_key:
                        matched_via = "base_random_key"
                    elif member_key and member_key:
                        matched_via = "member_random_key"
                    elif code_terms and any(
                        code in full_name.upper() for code in code_terms
                    ):
                        matched_via = "code_match"
                    elif name_terms and any(term in full_name for term in name_terms):
                        matched_via = "name_match"
                    elif args.product_name:
                        matched_via = "name_match"

                    results.append(
                        ProductMatch(
                            random_key=random_key,
                            persian_name=record[1],
                            english_name=record[2],
                            matched_via=matched_via,
                        )
                    )
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
