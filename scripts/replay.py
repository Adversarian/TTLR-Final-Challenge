"""Utility to inspect and replay logged conversations locally."""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx


def load_log(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda item: item.get("timestamp", ""))
    return records


def group_turns(records: Iterable[Dict[str, Any]], chat_id: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    current_messages: List[Dict[str, str]] = []
    for record in records:
        if record.get("chat_id") != chat_id:
            continue
        direction = record.get("direction")
        if direction == "incoming":
            current_messages.append(
                {
                    "type": record.get("message_type", "text"),
                    "content": record.get("content", ""),
                }
            )
        elif direction in {"outgoing", "system"}:
            if current_messages:
                turns.append({
                    "messages": current_messages,
                    "response": record.get("response"),
                    "direction": direction,
                })
                current_messages = []
    if current_messages:
        turns.append({"messages": current_messages, "response": None, "direction": "pending"})
    return turns


async def replay_http(turns: List[Dict[str, Any]], chat_id: str, endpoint: str) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        for turn in turns:
            payload = {"chat_id": chat_id, "messages": turn["messages"]}
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            print(f"Sent {payload} -> {response.json()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay conversations from log file")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("chat_id", help="Chat identifier to replay")
    parser.add_argument("--endpoint", help="Optional HTTP endpoint for live replay")
    args = parser.parse_args()

    records = load_log(args.logfile)
    turns = group_turns(records, args.chat_id)
    print(f"Loaded {len(turns)} turns for chat {args.chat_id}")
    for turn in turns:
        print("Messages:")
        for msg in turn["messages"]:
            print(f"  - ({msg['type']}) {msg['content']}")
        print(f"Response: {turn.get('response')}")
        print("---")

    if args.endpoint:
        asyncio.run(replay_http(turns, args.chat_id, args.endpoint))


if __name__ == "__main__":
    main()
