FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN python -m pip install --upgrade pip && python -m pip install uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY configs ./configs

RUN uv pip install --system ".[dev,local]"

COPY . .

CMD ["uvicorn", "graphrag_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
