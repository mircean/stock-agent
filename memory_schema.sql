-- Agent Memory Database Schema
-- Sections separated by "-- === VERSION N ===" markers
-- Version 0 = fresh database, Version 1+ = upgrades

-- === VERSION 0 ===
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_scores (
    date DATE,
    symbol TEXT,
    composite_score REAL,
    momentum_score REAL,
    quality_score REAL,
    technical_score REAL,
    current_price REAL,
    is_holding BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_agent_scores_date ON agent_scores(date DESC);
CREATE INDEX IF NOT EXISTS idx_agent_scores_symbol ON agent_scores(symbol, date DESC);

-- === VERSION 1 ===
-- Add run_time column for multiple runs per day
ALTER TABLE agent_scores ADD COLUMN run_time TEXT DEFAULT '00:00:00';