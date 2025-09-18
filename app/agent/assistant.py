from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel

from anyio import to_thread
from llama_index.agent.function_agent import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer, SimpleComposableMemory
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage as LlamaChatMessage, MessageRole
from llama_index.workflows import StartEvent, StopEvent, workflow, Context

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


class StructuredChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None


@lru_cache(maxsize=1)
def _vector_index() -> VectorStoreIndex:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be configured for vector search")

    parsed = urlparse(settings.database_url)
    query = parse_qs(parsed.query)
    store = PGVectorStore.from_params(
        database=parsed.path.lstrip("/"),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or query.get("user", [None])[0],
        password=parsed.password or query.get("password", [None])[0],
        table_name="product_embeddings",
        hybrid_search=True,
        text_search_config="simple",
    )
    return VectorStoreIndex.from_vector_store(store)


def _hybrid_retrieve(product_name: str, top_k: int) -> List[ProductMatch]:
    retriever = _vector_index().as_retriever(
        similarity_top_k=top_k,
        vector_store_kwargs={"hybrid_search": True},
    )
    nodes = retriever.retrieve(product_name)

    matches: List[ProductMatch] = []
    seen: set[str] = set()
    for node in nodes:
        random_key = node.metadata.get("random_key")
        if not random_key or random_key in seen:
            continue
        matches.append(
            ProductMatch(
                random_key=random_key,
                persian_name=node.metadata.get("persian_name"),
                english_name=node.metadata.get("english_name"),
                matched_via=node.metadata.get("match_type", "semantic"),
            )
        )
        seen.add(random_key)
    return matches


def _fetch_product_by_random_key(random_key: str) -> Optional[ProductMatch]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bp.random_key, bp.persian_name, bp.english_name
                FROM base_products bp
                WHERE bp.random_key = %s
                """,
                (random_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ProductMatch(
                random_key=row[0],
                persian_name=row[1],
                english_name=row[2],
                matched_via="base_random_key",
            )


def _fetch_product_by_member_key(member_key: str) -> Optional[ProductMatch]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bp.random_key, bp.persian_name, bp.english_name
                FROM members m
                JOIN base_products bp ON bp.random_key = m.base_random_key
                WHERE m.random_key = %s
                """,
                (member_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ProductMatch(
                random_key=row[0],
                persian_name=row[1],
                english_name=row[2],
                matched_via="member_random_key",
            )


@lru_cache(maxsize=1)
def get_lookup_tool() -> FunctionTool:
    def lookup_products(args: ProductLookupArgs) -> List[dict]:
        """Hybrid search over the product catalog."""
        matches: List[ProductMatch] = []
        seen: set[str] = set()

        if args.base_random_key:
            direct = _fetch_product_by_random_key(args.base_random_key.strip())
            if direct:
                matches.append(direct)
                seen.add(direct.random_key)

        if args.member_random_key:
            member = _fetch_product_by_member_key(args.member_random_key.strip())
            if member and member.random_key not in seen:
                matches.append(member)
                seen.add(member.random_key)

        if args.product_name:
            for match in _hybrid_retrieve(args.product_name, args.limit):
                if match.random_key in seen:
                    continue
                matches.append(match)
                seen.add(match.random_key)

        return [match.dict() for match in matches[: args.limit]]

    return FunctionTool.from_defaults(
        fn=lookup_products,
        name="lookup_products",
        description=(
            "Search the product catalog using hybrid semantic + lexical retrieval. "
            "Pass structured kwargs like product_name, base_random_key, member_random_key, limit."
        ),
    )


def build_agent(memory: SimpleComposableMemory) -> FunctionAgent:
    tool = get_lookup_tool()
    return FunctionAgent.from_tools(
        tools=[tool],
        system_prompt=(
            "You are Torob's shopping assistant. Use lookup_products to gather product candidates, "
            "then decide which random keys to return. For scenario-zero checks, respond 'pong' with null key lists "
            "when the user says 'ping', and echo requested random keys without validation. Maintain concise prose "
            "and never invent keys you did not retrieve."
        ),
        memory=memory,
    )


async def _execute_chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        return ChatResponse()

    memory = SimpleComposableMemory(
        primary_memory=ChatMemoryBuffer.from_defaults(token_limit=4000)
    )
    for message in request.messages[:-1]:
        role = MessageRole.ASSISTANT if getattr(message, "type", "user") == "assistant" else MessageRole.USER
        memory.put(LlamaChatMessage(role=role, content=message.content))

    agent = build_agent(memory)
    latest_message = request.messages[-1]
    parser = PydanticOutputParser(StructuredChatResponse)
    result = await to_thread.run_sync(
        agent.chat,
        latest_message.content,
        structured_output=parser,
    )
    if isinstance(result.output, StructuredChatResponse):
        structured = result.output
    else:
        structured = parser.parse(result.response)
    return ChatResponse(
        message=structured.message,
        base_random_keys=structured.base_random_keys,
        member_random_keys=structured.member_random_keys,
    )


class ChatWorkflowInput(StartEvent):
    request: ChatRequest


class ChatWorkflowOutput(StopEvent):
    response: ChatResponse


@workflow
async def chat_workflow(event: ChatWorkflowInput, _: Context) -> ChatWorkflowOutput:
    response = await _execute_chat(event.request)
    return ChatWorkflowOutput(response=response)


async def run_chat(request: ChatRequest) -> ChatResponse:
    result = await chat_workflow.run(ChatWorkflowInput(request=request))
    return result.response
