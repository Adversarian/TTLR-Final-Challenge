from app.models.chat import ChatRequest, ChatResponse


PING_CONTENT = "ping"
BASE_KEY_PREFIX = "return base random key:"
MEMBER_KEY_PREFIX = "return member random key:"


def handle_chat(request: ChatRequest) -> ChatResponse:
    """Handle scenario zero deterministic prompts without calling an LLM."""
    if not request.messages:
        return ChatResponse(message="", base_random_keys=None, member_random_keys=None)

    last_message = request.messages[-1].content.strip()

    if last_message == PING_CONTENT:
        return ChatResponse(message="pong")

    if last_message.startswith(BASE_KEY_PREFIX):
        key = last_message[len(BASE_KEY_PREFIX) :].strip()
        if key:
            return ChatResponse(base_random_keys=[key])
        return ChatResponse(base_random_keys=[])

    if last_message.startswith(MEMBER_KEY_PREFIX):
        key = last_message[len(MEMBER_KEY_PREFIX) :].strip()
        if key:
            return ChatResponse(member_random_keys=[key])
        return ChatResponse(member_random_keys=[])

    # Default empty payload keeps schema compliance for unsupported prompts.
    return ChatResponse()
