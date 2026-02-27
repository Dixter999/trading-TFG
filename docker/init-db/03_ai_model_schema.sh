#!/bin/bash
set -e

# Run the ai_model schema against the ai_model database (POSTGRES_DB).
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -d ai_model \
    -f /app/scripts/data/init_db.sql

echo "=== ai_model schema initialized ==="
