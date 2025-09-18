"""High-level orchestration for the Torob shopping assistant."""

from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage as LlamaChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer, SimpleComposableMemory
from llama_index.core.workflow import Context, workflow

from app.models.chat import ChatRequest, ChatResponse

from .models import (
    ChatWorkflowInput,
    ChatWorkflowOutput,
    StructuredChatResponse,
)
from .scenarios import handle_scenario_zero
from .tools import get_lookup_tool


def build_agent(_: SimpleComposableMemory | None = None) -> FunctionAgent:
    tool = get_lookup_tool()
    return FunctionAgent(
        tools=[tool],
        system_prompt=(
            "You are Torob's shopping assistant. For every user question about products, "
            "call lookup_products exactly once to gather product context (feature_list, seller_stats, offers) before answering. "
            "Use the returned data to answer succinctly in the user's language. "
            "When a query maps to one clear product, return a single base_random_key in base_random_keys. "
            "For attribute questions, quote the relevant feature value verbatim. "
            "For seller or price questions, respond with the numeric value from seller_stats.min_price or top_offers. "
            "Never invent random keys or make extra tool calls."
        ),
        output_cls=StructuredChatResponse,
        streaming=False,
    )


async def _execute_chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        return ChatResponse()

    latest_message = request.messages[-1]
    scenario_zero = handle_scenario_zero(latest_message.content)
    if scenario_zero is not None:
        return scenario_zero

    memory = SimpleComposableMemory(
        primary_memory=ChatMemoryBuffer.from_defaults(token_limit=4000)
    )
    chat_history: list[LlamaChatMessage] = []
    for message in request.messages[:-1]:
        llama_message = LlamaChatMessage(
            role=MessageRole.ASSISTANT if message.role == "assistant" else MessageRole.USER,
            content=message.content,
        )
        memory.put(llama_message)
        chat_history.append(llama_message)

    agent = build_agent()
    workflow_handler = agent.run(
        user_msg=latest_message.content,
        chat_history=chat_history,
        memory=memory,
    )
    workflow_result = await workflow_handler
    agent_result = getattr(workflow_result, "result", None)
    structured = None
    if agent_result is not None:
        structured = getattr(agent_result, "structured_response", None)
    if structured is None:
        structured = StructuredChatResponse()
    elif isinstance(structured, dict):
        structured = StructuredChatResponse(**structured)
    return ChatResponse(
        message=structured.message,
        base_random_keys=structured.base_random_keys,
        member_random_keys=structured.member_random_keys,
    )


@workflow
async def chat_workflow(event: ChatWorkflowInput, _: Context) -> ChatWorkflowOutput:
    response = await _execute_chat(event.request)
    return ChatWorkflowOutput(response=response)


async def run_chat(request: ChatRequest) -> ChatResponse:
    result = await chat_workflow.run(ChatWorkflowInput(request=request))
    return result.response


__all__ = ["build_agent", "chat_workflow", "run_chat"]
