FROM python:3.11-slim

WORKDIR /app

# System dependencies for psycopg2 and scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Default: idle shell (overridden by docker-compose command)
CMD ["bash"]
