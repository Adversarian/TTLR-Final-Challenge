# Entry #1: Kickoff Alignment
- Reviewed the full problem statement to extract dataset schema, scenario expectations, and response format.
- Logged non-negotiable operating rules and open technical questions in `AGENTS.md` for ongoing reference.
- Identified critical early design concerns: conversation replay tooling, multimodal support, and data ingestion strategy.

# Entry #2: Runtime Strategy Checkpoint
- Clarified latency and budget constraints: target <30s total per request, minimize LLM usage via short prompts and limited tool calls.
- Decided to prioritize SQL-based retrieval over broad RAG; vector search (Qdrant) will only join if similarity matching becomes essential.
- Established cautious emission rule for product/member keys to avoid premature scenario termination in multi-turn flows.
- Added plan to tier LLM models by scenario criticality to balance accuracy with token spend.
