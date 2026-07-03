FROM python:3.11-slim

# System deps for chromadb + networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user (UID 1000, per HuggingFace Spaces convention)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN chown appuser:appuser /app
USER appuser

# Pre-download fastembed model as the runtime user so first ingest is instant
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Copy application code
COPY --chown=appuser:appuser . .

# Create data dirs (ephemeral on free tier)
RUN mkdir -p /app/chroma_data

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# --proxy-headers so the rate limiter sees real client IPs behind the Spaces proxy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
