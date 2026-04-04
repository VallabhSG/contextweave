FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb + networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed model so first ingest is instant
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Copy application code
COPY . .

# Create data dirs (ephemeral on free tier)
RUN mkdir -p /app/chroma_data

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
