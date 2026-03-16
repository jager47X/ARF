FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# Validate config on build
RUN python config_schema.py || true

CMD ["python", "-m", "pytest", "tests/", "-v"]
