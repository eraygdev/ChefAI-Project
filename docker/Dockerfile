# ============================================================
# STAGE 1: Builder
# ============================================================
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

# Sistem bagimliliklari (build icin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry kur
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /build

# Sadece pyproject + lock kopyala (cache icin)
COPY pyproject.toml poetry.lock ./

# requirements.txt'e export et (dev haric)
RUN poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt

# ============================================================
# STAGE 2: Production
# ============================================================
FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Builder'dan requirements.txt kopyala
COPY --from=builder /build/requirements.txt .

# Bagimliliklari kur
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kodu kopyala
COPY src/ ./src/
COPY README.md ./

# Port
EXPOSE 8000

# Saglik kontrolu
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Gunicorn + Uvicorn workers (PDF sayfa 18)
CMD ["gunicorn", \
    "src.api.main:app", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000"]
