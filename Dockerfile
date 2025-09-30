# syntax=docker/dockerfile:1

# Use a slim Python base image compatible with TensorFlow CPU
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# System deps: build essentials and runtime libs for Pillow/TensorFlow
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1 \
       libglib2.0-0 \
       libjpeg62-turbo \
       zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the port Render assigns (Render sets $PORT at runtime)
EXPOSE 8000

# Healthcheck: ping the root path
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request, os; \
  urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/').read()" || exit 1

# Use environment PORT if provided by Render; default to 8000 locally
ENV PORT=8000

# Start the FastAPI app with uvicorn
CMD ["bash", "-lc", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
