# Use a stable python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MONGO_URI=mongodb://mongodb:27017/stock_analytics

# Set work directory
WORKDIR /app

# Install system dependencies (needed for compiling some python wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage docker cache
COPY requirements.txt /app/

# Install Python dependencies. The generic requirements file includes GPU-enabled
# packages by default; this VM deployment only needs CPU inference/training.
RUN sed \
        -e '/^[[:space:]]*torch[<>=!~]/d' \
        -e '/^[[:space:]]*xgboost[<>=!~]/d' \
        requirements.txt > /tmp/requirements.runtime.txt \
    && pip install --no-cache-dir -r /tmp/requirements.runtime.txt \
        filelock fsspec networkx sympy \
        xgboost-cpu==3.2.0 \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --no-deps torch==2.13.0

# Copy the rest of the application code
COPY . /app/

# Expose port
EXPOSE 8000

# Basic liveness check for reverse proxies/orchestrators.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS --max-time 8 http://127.0.0.1:8000/api/control/health > /dev/null || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
