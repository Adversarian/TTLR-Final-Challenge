# TTLR Final Challenge
"Confuse yourself, confuse the enemy." - *Confucius*

Torob Turbo LLM Rush Finals: Submission from Arian Tashakkor.


# Tech Stack
- **AI Orchestration**: Pydantic-AI
- **DB and Vector DB**: PostgreSQL + PGVector extension
- **DB Ops**: SQLAlchemy + Alembic
- **Observability**: Logfire (+ manual logging)
- **Serving**: FastAPI
- **LLM Vendor**: OpenAI
- **Image Embedding** DINOv2-base @ 768 by Facebook Research. Deployed [here](https://github.com/Adversarian/TTLR-DINOv2-Embedding-Service).

# Agents
1. **Default Agent [GPT-4.1]** (under `app/[factory|prompts|schemas|tools].py`): Responsible for scenarios 1, 2, 3 and 5. 
2. **Multi-turn Agent [GPT-4.1]** (under `app/multiturn/[factory|prompts|schemas|state|tools].py`): Responsible for handling scenario 4.
3. **Single/Multi-turn Router [GPT-4.1-Mini]** (under `app/router/[factory|prompts|schemas|state].py`): Routes incoming user queries to either the `Default Agent` for single-turn textual scenarios(1, 2, 3 and 5) and to `Multi-turn Agent` for scenario 4, the only multi-turn scenario (the basis for which can be found under `app/router/prompts.py`). The routing decisions are cached by `chat_id` so we don't unnecessarily call the router on subsequent turns of the same conversation, not opening ourselves up to potential false negatives as well as cutting down on latency and token expenditure.
4. **Image Agent [GPT-4.1]** (under `app/image/[factory|prompts].py`): A very simple agent used for describing images. Used for scenario 6.
5. **Vision Router [GPT-4.1-Mini]** (under `app/vision_router/[factory|prompts|schemas].py`): Acts in API layer if an `image` component is detected in the incoming user message. Takes the text part of the message to either route it to the `Image Agent` for scenario 6 or let the logic for scenario 7 handle it.

# Quick Scenario Rundown
## Scenario 0
Connection test. Handled statically.
## Scenario 1
`Default Agent` solves this with one call to `PRODUCT_SEARCH_TOOL`.
## Scenario 2
`Default Agent` solves this with a `PRODUCT_SEARCH_TOOL -> FEATURE_LOOKUP_TOOL` chain.
## Scenario 3
`Default Agent` solves this with a `PRODUCT_SEARCH_TOOL -> SELLER_STATISTICS_TOOL` chain.
## Scenario 4
`Multi-turn Agent` asks two generalist questions regarding product and shop in two turns. On subsequent turns it keeps calling `SEARCH_MEMBERS_TOOL` to refine it's list of potential member candidates. If at any point in time the list of candidates falls below 5 members, it presents them to the user and prompts them to choose one.
## Scenario 5
Given $N$ products for comparison: `Defaults Agent` solves this by performing $N \times$`PRODUCT_SEARCH_TOOL ->` ($N \times$`FEATURE_LOOKUP_TOOL` | $N \times$ `SELLER_STATISTICS_TOOL`) parallel tool calls.
## Scenario 6
Simple prompt and image are given to a multi-modal LLM, results are returned verbatim (This was mostly a simple prompt tuning task).
## Scenario 7
The image is sent to an [embedding service deployed separately](https://github.com/Adversarian/TTLR-DINOv2-Embedding-Service). The top 1 closest image's attached `base_random_key` is returned as output.


