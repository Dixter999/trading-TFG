#!/bin/bash
set -e

# Create the 'markets' database (ai_model is already created as POSTGRES_DB).
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -d "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE markets;
    GRANT ALL PRIVILEGES ON DATABASE markets TO "$POSTGRES_USER";
EOSQL

echo "=== Created 'markets' database ==="
