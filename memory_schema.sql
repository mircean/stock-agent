-- Agent Memory Database Schema
-- Stores daily stock scores from trading analysis

CREATE TABLE IF NOT EXISTS agent_scores (
    date DATE,
    symbol TEXT,
    composite_score REAL,
    momentum_score REAL,
    quality_score REAL,
    technical_score REAL,
    current_price REAL,
    is_holding BOOLEAN,
    PRIMARY KEY (date, symbol)
);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_agent_scores_date ON agent_scores(date DESC);

-- Index for symbol lookups
CREATE INDEX IF NOT EXISTS idx_agent_scores_symbol ON agent_scores(symbol, date DESC);