# ============================================================
# ChefAI API Image (Standalone CPU Inference)
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image lightweight
RUN pip install --no-cache-dir \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cpu

# Install API and inference dependencies
RUN pip install --no-cache-dir \
    "timm>=1.0.30,<2.0.0" \
    "numpy>=1.26.4,<2.0" \
    "pillow>=10.4.0,<11.0" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.34" \
    "gunicorn>=23.0" \
    "python-multipart>=0.0.20" \
    "pydantic>=2.10" \
    "python-dotenv>=1.0" \
    "prometheus-fastapi-instrumentator>=7.0" \
    "prometheus-client>=0.21"

COPY src/ ./src/
COPY README.md ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", \
    "src.api.main:app", \
    "--chdir", "/app", \
    "--pythonpath", "/app", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000"]
