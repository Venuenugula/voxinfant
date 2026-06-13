# VoxInfant — single-container deploy (FastAPI + static UI + baked-in model).
FROM python:3.10-slim

# System libs needed by librosa / soundfile (audio I/O + resampling).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code + trained artifacts + UI.
COPY src/ ./src/
COPY api/ ./api/
COPY web/ ./web/
COPY models/ ./models/
COPY config.yaml ./

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Honour $PORT (Railway/Render set it); default 8000 locally.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
