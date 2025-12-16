-- NASDAQ Stock Database Schema
-- Creates tables for comprehensive stock data storage

-- Stocks metadata table
CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily stock prices and metrics
CREATE TABLE IF NOT EXISTS stock_prices (
    symbol TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

-- Stock fundamentals history (historical with dates)
CREATE TABLE IF NOT EXISTS stock_fundamentals_history (
    symbol TEXT,
    date DATE,
    market_cap REAL,
    enterprise_value REAL,
    pe_ratio REAL,
    peg_ratio REAL,
    price_to_book REAL,
    price_to_sales REAL,
    ev_to_revenue REAL,
    ev_to_ebitda REAL,
    debt_to_equity REAL,
    return_on_equity REAL,
    return_on_assets REAL,
    gross_margin REAL,
    operating_margin REAL,
    profit_margin REAL,
    beta REAL,
    dividend_yield REAL,
    payout_ratio REAL,
    shares_outstanding REAL,
    float_shares REAL,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

-- Stock fundamentals view (latest snapshot only, for agent queries)
CREATE VIEW IF NOT EXISTS stock_fundamentals AS
SELECT symbol, market_cap, enterprise_value, pe_ratio, peg_ratio, price_to_book,
       price_to_sales, ev_to_revenue, ev_to_ebitda, debt_to_equity, return_on_equity,
       return_on_assets, gross_margin, operating_margin, profit_margin, beta,
       dividend_yield, payout_ratio, shares_outstanding, float_shares
FROM stock_fundamentals_history
WHERE date = (SELECT MAX(date) FROM stock_fundamentals_history);

-- Stock splits and dividends
CREATE TABLE IF NOT EXISTS stock_actions (
    symbol TEXT,
    date DATE,
    action_type TEXT CHECK(action_type IN ('split', 'dividend')),
    value REAL,
    ratio TEXT,
    PRIMARY KEY (symbol, date, action_type),
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

-- Stock statistics history (historical with dates)
CREATE TABLE IF NOT EXISTS stock_statistics_history (
    symbol TEXT,
    date DATE,
    fifty_two_week_high REAL,
    fifty_two_week_low REAL,
    fifty_day_average REAL,
    two_hundred_day_average REAL,
    avg_volume_10day INTEGER,
    avg_volume_3month INTEGER,
    shares_short REAL,
    short_ratio REAL,
    short_percent_float REAL,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

-- Stock statistics view (latest snapshot only, for agent queries)
CREATE VIEW IF NOT EXISTS stock_statistics AS
SELECT symbol, fifty_two_week_high, fifty_two_week_low, fifty_day_average,
       two_hundred_day_average, avg_volume_10day, avg_volume_3month, shares_short,
       short_ratio, short_percent_float
FROM stock_statistics_history
WHERE date = (SELECT MAX(date) FROM stock_statistics_history);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_history_symbol_date ON stock_fundamentals_history(symbol, date);
CREATE INDEX IF NOT EXISTS idx_stock_actions_symbol_date ON stock_actions(symbol, date);
CREATE INDEX IF NOT EXISTS idx_stock_statistics_history_symbol_date ON stock_statistics_history(symbol, date);