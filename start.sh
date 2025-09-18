#!/usr/bin/env bash
set -euo pipefail

: "${APP_HOST:=0.0.0.0}"
: "${APP_PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${TOROB_BOOTSTRAP:=0}"
: "${TOROB_DATA_DIR:=/data/torob}"
: "${TOROB_AUTO_EMBED:=0}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ "$TOROB_BOOTSTRAP" = "1" ]; then
  ARCHIVE_PATH="$TOROB_DATA_DIR/torob-turbo-stage2.tar.gz"
  RAW_DIR="$TOROB_DATA_DIR/raw"
  PARQUET_DIR_DEFAULT="$RAW_DIR/torob-turbo-stage2"
  PARQUET_DIR="${TOROB_PARQUET_DIR:-$PARQUET_DIR_DEFAULT}"

  mkdir -p "$TOROB_DATA_DIR"

  if [ ! -d "$RAW_DIR" ] || { [ ! -d "$PARQUET_DIR" ] && ! compgen -G "$RAW_DIR/*.parquet" > /dev/null; }; then
    echo "[bootstrap] Downloading dataset to $ARCHIVE_PATH"
    uv run python -m scripts.download_data \
      --drive-id "${TOROB_DRIVE_ID:?TOROB_DRIVE_ID must be set when TOROB_BOOTSTRAP=1}" \
      --output "$ARCHIVE_PATH" \
      --extract-dir "$RAW_DIR" \
      --overwrite
  else
    echo "[bootstrap] Parquet directory already present: $PARQUET_DIR"
  fi

  # Determine actual parquet directory (allow override via TOROB_PARQUET_DIR)
  if [ ! -d "$PARQUET_DIR" ]; then
    if compgen -G "$RAW_DIR/*.parquet" > /dev/null; then
      PARQUET_DIR="$RAW_DIR"
    else
      FIRST_SUBDIR=$(find "$RAW_DIR" -maxdepth 1 -mindepth 1 -type d | head -n 1)
      if [ -n "$FIRST_SUBDIR" ]; then
        PARQUET_DIR="$FIRST_SUBDIR"
      fi
    fi
  fi

  if [ -n "${DATABASE_URL:-}" ]; then
    if [ -d "$PARQUET_DIR" ]; then
      echo "[bootstrap] Ingesting parquet files into database from $PARQUET_DIR"
      uv run python -m scripts.ingest \
        --data-dir "$PARQUET_DIR" \
        --database-url "$DATABASE_URL" \
        --if-exists replace
    else
      echo "[bootstrap] ERROR: Could not locate parquet directory under $RAW_DIR" >&2
      exit 1
    fi
  else
    echo "[bootstrap] DATABASE_URL not set; skipping ingestion"
  fi

  if [ "$TOROB_AUTO_EMBED" = "1" ]; then
    if [ -z "${DATABASE_URL:-}" ]; then
      echo "[bootstrap] DATABASE_URL not set; cannot build embeddings" >&2
    elif [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "[bootstrap] OPENAI_API_KEY not set; skipping embedding generation" >&2
    else
      echo "[bootstrap] Generating product embeddings"
      uv run python -m scripts.embed_products \
        --database-url "$DATABASE_URL" \
        --model "${OPENAI_EMBED_MODEL:-text-embedding-3-large}"
    fi
  fi
fi

echo "Starting FastAPI server"
exec uv run python -m app.server
