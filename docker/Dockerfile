# Multi-stage build
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

# Sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Poetry kur
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Bagimliliklari once kopyala (cache icin)
COPY pyproject.toml poetry.lock ./

# Bagimliliklari kur (dev grubu haric)
RUN poetry install --no-root --without dev

# Kaynak kodu kopyala
COPY src/ ./src/
COPY README.md ./

# Root paketi kur
RUN poetry install --only-root

# Port
EXPOSE 8000

# Saglik kontrolu
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Varsayilan komut (API calistir)
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
