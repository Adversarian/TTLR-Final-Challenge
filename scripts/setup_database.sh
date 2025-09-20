#!/bin/sh
set -eu

RETRIES=${DB_BOOT_RETRIES:-10}
SLEEP_SECONDS=${DB_BOOT_SLEEP_SECONDS:-3}
ATTEMPT=1

until uv run python -m app.scripts.prepare_database; do
  if [ "$ATTEMPT" -ge "$RETRIES" ]; then
    echo "Database preparation failed after ${RETRIES} attempts" >&2
    exit 1
  fi
  ATTEMPT=$((ATTEMPT + 1))
  echo "Database not ready yet, retrying in ${SLEEP_SECONDS}s (attempt ${ATTEMPT}/${RETRIES})"
  sleep "$SLEEP_SECONDS"
done

exec "$@"
