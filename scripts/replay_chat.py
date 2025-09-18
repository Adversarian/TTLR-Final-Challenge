"""Replay logged chat transcripts locally or against a running API."""

import argparse
import json
import os
import sys
from urllib import error, request

from app.services.replay import load_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat_id", help="Chat identifier to replay")
    parser.add_argument(
        "--log-dir",
        default=os.getenv("REPLAY_LOG_DIR", "./logs/replay"),
        help="Directory containing JSONL replay logs.",
    )
    parser.add_argument(
        "--endpoint",
        help="Optional /chat endpoint to POST each recorded request for re-simulation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the requests without posting them to the endpoint.",
    )
    return parser.parse_args()


def _post(endpoint: str, payload: dict) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with request.urlopen(req) as resp:
        charset = resp.headers.get_content_charset("utf-8")
        response_body = resp.read().decode(charset)
        return resp.status, response_body


def main() -> int:
    args = parse_args()
    entries = load_conversation(args.chat_id, args.log_dir)
    if not entries:
        print(
            f"No conversation found for chat_id={args.chat_id} in {args.log_dir}",
            file=sys.stderr,
        )
        return 1

    for entry in entries:
        timestamp = entry.get("timestamp")
        payload = entry.get("request") or {}
        response = entry.get("response") or {}
        print(f"\n=== {timestamp} ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("--- response ---")
        print(json.dumps(response, ensure_ascii=False, indent=2))

        if args.endpoint:
            if args.dry_run:
                print(f"[dry-run] would POST to {args.endpoint}")
            else:
                try:
                    status, body = _post(args.endpoint, payload)
                except error.HTTPError as exc:
                    print(
                        f"[endpoint {exc.code}] {exc.read().decode('utf-8', errors='ignore')}"
                    )
                except error.URLError as exc:
                    print(f"[endpoint error] {exc}")
                else:
                    print(f"[endpoint {status}] {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
