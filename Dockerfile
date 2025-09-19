FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH" \
    DATA_URL="https://drive.google.com/uc?id=1W4mSI33IbeKkWztK3XmE05F7m4tNYDYu"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system .

COPY . .

EXPOSE 8000

CMD ["bash", "-c", "python -m app.ingestion --database ${DATABASE_URL} --url ${DATA_URL} && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
