# Agent Guidelines

## Current implementation
- FastAPI `/chat` endpoint handles the static sanity checks ("ping", base key echo, member key echo) before delegating to the agent.
- A Pydantic-AI agent instrumented with Logfire resolves catalogue lookups through PostgreSQL fuzzy search and feature extraction tools.
- Fuzzy catalogue search collects trigram-matched candidates and reranks them with PostgreSQL full-text `ts_rank_cd` scoring derived from a generated search vector so TF/IDF precision wins.
- Feature extraction returns the entire flattened feature list for a base product so the agent can pick the correct attribute without repeated tool calls.
- Seller-focused questions are handled via a `get_seller_statistics` tool that joins members, shops, and cities to compute price, score, warranty, and availability aggregates while feeding a numeric answer back to the HTTP layer.
- The data layer assumes PostgreSQL connection details are provided via the environment variables listed in `.env.template`.
- The monolithic `app/agent.py` module has been decomposed into an `app/agent/` package with dedicated files for dependencies, schemas, tools, prompts, and the agent factory to keep the codebase maintainable.
- The system prompt now teaches the model how to compare multiple candidate products in one turn while still returning a single base key.
- The OpenAI client enables `parallel_tool_calls` so the runtime can execute independent catalogue lookups concurrently when the model requests it.
- Base product ingestion now preserves the raw list of member random keys from the dataset in addition to the normalised `members` table for parity with the published schema.
- Prompting updates steer the agent to craft richer product search queries, avoid duplicate tool calls, and finish once confident instead of looping on identical tool invocations.
- Image traffic is routed to a dedicated vision agent that consumes the uploaded BinaryContent directly and answers with a few Persian words describing the dominant object, without invoking catalogue tools.
- The `/chat` endpoint treats the incoming `messages` array as the modalities of a single user turn; the presence of any `image` part triggers the vision agent even if the final element is textual.
- Vision inference reuses the `OPENAI_MODEL` configuration through Pydantic-AI's multimodal support, so no separate vision-specific environment variables are required.
- Turn routing uses a lightweight `shopping-turn-router` agent powered by `OPENAI_ROUTER_MODEL` to label each text-only request as `single_turn` or `multi_turn` immediately after the vision hand-off check.
- The multi-turn flow shares compact conversation state via `app/agent/multiturn/`. `MemberDetails` captures hard filters (brand/category/city/price/warranty/score) plus fuzzy keywords and product attributes, while `TurnState` tracks turn counts, candidate previews, and optional stop reasons.
- Multi-turn NLU first attempts an LLM-backed extractor (`multiturn-nlu-extractor`) that produces a `MemberDelta` payload directly from the utterance; if credentials are missing or the run fails, it falls back to the deterministic regex/keyword heuristic used previously. The extractor is instrumented via Logfire and its model can be configured with `OPENAI_MULTITURN_NLU_MODEL`.
- Candidate search for the multi-turn flow lives in `app/agent/multiturn/search.py`. It executes a single PostgreSQL query per invocation to compute the total match count, top five ranked offers, and capped distributions for brand, city, price bands (500,000 Tomans per bucket), and warranty preference. Ranking blends TF/IDF on `search_vector`, trigram similarity over names and features, and a small seller-score boost.
- Multi-turn helpers avoid bespoke Logfire spans; only agent factories call `_ensure_logfire()` so instrumentation mirrors the single-turn module.
- Conversation memory persists only the structured `TurnState` plus a ≤200 character summary and keeps trace payloads compact without emitting bespoke Logfire spans.
- The multi-turn dialogue policy runs through `MultiTurnCoordinator`, which keeps state in `ConversationMemory`, logs each turn via Logfire, and forgets the chat once a stop reason is reached.
- The policy first tries to resolve option selections, otherwise applies the NLU delta, executes one candidate search, and, when necessary, relaxes constraints in the following order: keywords → product attributes → price range → brand → city → category → minimum score → warranty.
- When multiple candidates remain the coordinator asks at most two clarifying questions (`product_scope` then `shop_scope`) before presenting compact options; once options are shown it expects numeric replies (Persian or ASCII digits) and replays the option list if the response is invalid.
- The multi-turn coordinator serialises access per `chat_id`, caches the last successful numeric selection for 60 seconds so repeated replies return the same `member_random_key`, and wraps each policy turn in a 25s timeout that falls back to short Persian error/timeout messages while preserving the previous state snapshot.

## Database indexes
- `base_products`
  - `idx_base_products_persian_name_trgm` (GIN with `gin_trgm_ops`) accelerates fuzzy name lookups for the `search_base_products` tool.
  - `idx_base_products_english_name_trgm` (GIN with `gin_trgm_ops`) backs English-name fuzzy searches for the same tool.
  - `idx_base_products_search_vector` (GIN on a generated `tsvector`) supports TF/IDF reranking without full table scans.
  - `idx_base_products_category` and `idx_base_products_brand` remain available for potential category/brand filters while exploring catalogue data.
- `members`
  - `idx_members_base_random_key` ensures the seller statistics aggregation can quickly collect offers for a base product.
  - `idx_members_shop_id` keeps lookups by shop efficient for warranty/score joins.

## Ground rules for new changes
- Keep solutions simple, well-documented, and strongly typed; prefer the minimal implementation that satisfies the competition scenarios without per-scenario branching (scenario 0 may remain hard-coded).
- Use `uv` for dependency management, FastAPI for the HTTP layer, and Pydantic-AI for agent workflows. Avoid introducing conflicting or deprecated libraries.
- When modifying agent behaviour, express shared heuristics in tool descriptions or the system prompt rather than hard-coding logic paths per scenario.
- Seller statistics must continue to populate `numeric_answer` so that the `/chat` endpoint can enforce digit-only responses for competition checks.
- Maintain parity across scenarios when enhancing prompts—new rules should describe general behaviours (e.g., comparisons, confidence handling) instead of referencing specific test IDs.
- When handling images, keep the response to a single concise sentence or noun phrase, default to Persian phrasing, and avoid returning product keys while the vision agent is limited to dominant-object descriptions.
- Run `uv run pytest` (and any other affected checks) before completing a task to keep the test suite passing.
- Update this file whenever project rules or capabilities change so future tasks inherit accurate guidance.
- Keep tool descriptions aligned with their real capabilities (search terms can include distinctive attributes; seller statistics accepts only the base random key with an optional city filter and returns the full aggregate payload).
- Enclose every `agent.run` invocation in the shared `_run_agent_with_retry` helper so retries remain consistent across the API.
- The router must respond with a minimal JSON object (`{"mode": "single_turn"}` or `{"mode": "multi_turn"}`) and should fail closed to the single-turn flow if routing encounters an error.
