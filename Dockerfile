FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

COPY . .

# Validate config on build
RUN python config_schema.py || true

# Run lint check (fail build on lint errors)
RUN python -m ruff check config.py config_schema.py rag_dependencies/ tests/

CMD ["python", "-m", "pytest", "tests/", "-v"]
