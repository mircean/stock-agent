-- Portfolio Database Schema
-- Tracks historical portfolio performance for charting and analysis
-- Current portfolio state is stored in portfolio.json

-- Daily portfolio values for historical charting and performance tracking
CREATE TABLE IF NOT EXISTS portfolio_history (
    date DATE PRIMARY KEY,
    cash NUMERIC(12,4) NOT NULL,
    positions_value NUMERIC(12,4) NOT NULL,
    total_value NUMERIC(12,4) NOT NULL,
    position_count INTEGER                      -- Number of positions held
);

CREATE INDEX IF NOT EXISTS idx_portfolio_history_date ON portfolio_history(date DESC);

-- NASDAQ 100 index historical values for benchmarking
CREATE TABLE IF NOT EXISTS nasdaq100_history (
    date DATE PRIMARY KEY,
    value NUMERIC(12,4) NOT NULL                -- NDX closing value
);

CREATE INDEX IF NOT EXISTS idx_nasdaq100_history_date ON nasdaq100_history(date DESC);
