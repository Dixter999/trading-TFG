# Technical Indicators Query Performance Guide

## Table: `technical_indicators`

### Index Strategy

The `technical_indicators` table uses a carefully designed index strategy optimized for time-series queries:

#### 1. Primary Key: `(symbol, timeframe, timestamp)`
- **Purpose**: Ensures uniqueness of each indicator record
- **Type**: B-tree index (automatically created)
- **Use Case**: Point lookups for specific symbol+timeframe+timestamp combinations
- **Query Pattern**:
  ```sql
  SELECT * FROM technical_indicators
  WHERE symbol = 'EURUSD' AND timeframe = 'H1' AND timestamp = '2024-01-15 14:00:00+00';
  ```

#### 2. Index: `idx_technical_indicators_timestamp` (timestamp DESC)
- **Purpose**: Fast retrieval of latest indicators across all symbols
- **Type**: B-tree index, descending order
- **Use Case**: Getting most recent indicators without symbol filter
- **Query Pattern**:
  ```sql
  SELECT * FROM technical_indicators
  ORDER BY timestamp DESC
  LIMIT 100;
  ```
- **Performance**: O(log n) for latest records retrieval

#### 3. Index: `idx_technical_indicators_symbol_timeframe` (symbol, timeframe)
- **Purpose**: Fast filtering by symbol and timeframe
- **Type**: Composite B-tree index
- **Use Case**: Getting all indicators for a specific symbol/timeframe pair
- **Query Pattern**:
  ```sql
  SELECT * FROM technical_indicators
  WHERE symbol = 'EURUSD' AND timeframe = 'H1';
  ```
- **Performance**: O(log n) for symbol+timeframe filtering

#### 4. Index: `idx_technical_indicators_lookup` (symbol, timeframe, timestamp DESC)
- **Purpose**: Optimized for the most common query pattern
- **Type**: Composite B-tree index, timestamp in descending order
- **Use Case**: Getting recent indicators for specific symbol/timeframe
- **Query Pattern**:
  ```sql
  SELECT * FROM technical_indicators
  WHERE symbol = 'EURUSD' AND timeframe = 'H1'
  ORDER BY timestamp DESC
  LIMIT 20;
  ```
- **Performance**: O(log n) + k (where k = LIMIT), uses index-only scan

## Optimal Query Patterns

### ✅ Fast Queries (Use Indexes Efficiently)

#### 1. Get Latest N Indicators for Symbol/Timeframe
```sql
SELECT * FROM technical_indicators
WHERE symbol = :symbol AND timeframe = :timeframe
ORDER BY timestamp DESC
LIMIT :n;
```
**Index Used**: `idx_technical_indicators_lookup`

#### 2. Get Indicators in Time Range for Symbol/Timeframe
```sql
SELECT * FROM technical_indicators
WHERE symbol = :symbol
  AND timeframe = :timeframe
  AND timestamp BETWEEN :start_time AND :end_time
ORDER BY timestamp DESC;
```
**Index Used**: `idx_technical_indicators_lookup`

#### 3. Get Latest Indicators Across All Symbols
```sql
SELECT * FROM technical_indicators
ORDER BY timestamp DESC
LIMIT :n;
```
**Index Used**: `idx_technical_indicators_timestamp`

#### 4. Point Lookup (Exact Match)
```sql
SELECT * FROM technical_indicators
WHERE symbol = :symbol
  AND timeframe = :timeframe
  AND timestamp = :exact_time;
```
**Index Used**: Primary Key

### ⚠️ Slow Queries (Avoid These Patterns)

#### 1. Filtering by Indicator Values (No Index)
```sql
-- SLOW: No index on rsi_14
SELECT * FROM technical_indicators
WHERE rsi_14 > 70;
```
**Solution**: Add specific index if this query is frequent, or filter in application layer after retrieving data.

#### 2. Using Functions on Indexed Columns
```sql
-- SLOW: Function on timestamp prevents index usage
SELECT * FROM technical_indicators
WHERE DATE(timestamp) = '2024-01-15';
```
**Solution**: Use range queries instead:
```sql
SELECT * FROM technical_indicators
WHERE timestamp >= '2024-01-15 00:00:00+00'
  AND timestamp < '2024-01-16 00:00:00+00';
```

#### 3. OR Conditions on Different Columns
```sql
-- SLOW: Cannot use composite index efficiently
SELECT * FROM technical_indicators
WHERE symbol = 'EURUSD' OR timeframe = 'H1';
```
**Solution**: Use UNION or separate queries.

## Index Maintenance

### Monitoring Index Usage
```sql
-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'technical_indicators'
ORDER BY idx_scan DESC;
```

### Checking Index Bloat
```sql
-- Estimate index bloat
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE tablename = 'technical_indicators';
```

### REINDEX (When Needed)
```sql
-- Rebuild indexes if bloated (requires table lock)
REINDEX TABLE technical_indicators;

-- Or rebuild specific index
REINDEX INDEX idx_technical_indicators_lookup;
```

## Query Plan Analysis

Use `EXPLAIN ANALYZE` to verify index usage:

```sql
EXPLAIN ANALYZE
SELECT * FROM technical_indicators
WHERE symbol = 'EURUSD' AND timeframe = 'H1'
ORDER BY timestamp DESC
LIMIT 20;
```

**Look for**:
- `Index Scan` or `Index Only Scan` (good)
- `Bitmap Index Scan` (acceptable for medium result sets)
- `Seq Scan` (bad - means no index used)

## Performance Benchmarks

Based on table size:

| Rows | Query Type | Index Used | Expected Time |
|------|-----------|-----------|---------------|
| 1K | Point lookup | Primary Key | < 1ms |
| 1K | Latest 20 for symbol | Composite | < 1ms |
| 100K | Point lookup | Primary Key | < 2ms |
| 100K | Latest 20 for symbol | Composite | < 5ms |
| 1M | Point lookup | Primary Key | < 5ms |
| 1M | Latest 20 for symbol | Composite | < 10ms |
| 10M | Point lookup | Primary Key | < 10ms |
| 10M | Latest 20 for symbol | Composite | < 20ms |

## Index Size Estimates

- Primary Key: ~40% of table size
- Each additional index: ~30% of table size
- Total index overhead: ~130-150% of table size

**Example**: 1GB table → ~1.3-1.5GB of indexes

## Recommendations

1. **Monitor**: Track query performance and index usage regularly
2. **Vacuum**: Run `VACUUM ANALYZE` after bulk inserts
3. **Partition**: Consider partitioning by timeframe or date when table exceeds 10M rows
4. **Archive**: Move old data to separate table/partition
5. **Cache**: Use application-level caching for frequently accessed indicators
