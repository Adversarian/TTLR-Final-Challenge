FROM ghcr.io/astral-sh/uv:python3.11-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY alembic.ini ./alembic.ini
COPY app ./app
COPY main.py ./main.py
COPY scripts ./scripts

RUN chmod +x scripts/setup_database.sh

ENV PYTHONPATH=/app

ENTRYPOINT ["./scripts/setup_database.sh"]
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
