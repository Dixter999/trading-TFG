#!/bin/bash
set -e

# Create per-symbol per-timeframe rate tables in the 'markets' database.
# These tables mirror the production PostgreSQL structure.

SYMBOLS="eurusd gbpusd usdjpy eurjpy usdcad eurcad usdchf eurgbp"
TIMEFRAMES="m30 h1 h2 h3 h4 h6 h8 h12 d1"

for symbol in $SYMBOLS; do
    for tf in $TIMEFRAMES; do
        TABLE="${symbol}_${tf}_rates"
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -d markets <<-EOSQL
            CREATE TABLE IF NOT EXISTS ${TABLE} (
                rate_time       BIGINT PRIMARY KEY,
                open            DECIMAL(18, 8) NOT NULL,
                high            DECIMAL(18, 8) NOT NULL,
                low             DECIMAL(18, 8) NOT NULL,
                close           DECIMAL(18, 8) NOT NULL,
                volume          DECIMAL(18, 8) NOT NULL,
                readable_date   TIMESTAMP WITHOUT TIME ZONE
            );
            CREATE INDEX IF NOT EXISTS idx_${symbol}_${tf}_time
                ON ${TABLE} (rate_time DESC);
EOSQL
    done
done

echo "=== Markets schema created ($(echo $SYMBOLS | wc -w) symbols x $(echo $TIMEFRAMES | wc -w) timeframes) ==="
