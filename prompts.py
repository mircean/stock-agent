"""
Prompts for the Stock Trading Agent
"""

import config


def get_database_schema() -> str:
    """
    Get the database schema documentation.

    Returns:
        Database schema documentation string
    """
    return """# NASDAQ Stock Database Schema

You have access to a comprehensive NASDAQ 100 stock database with 3 years of historical data. The database contains the following tables:

## Tables Overview

### 1. `stocks` - Stock metadata
Primary table containing basic information about each stock.

**Schema:**
```sql
CREATE TABLE stocks (
    symbol TEXT PRIMARY KEY,        -- Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    name TEXT,                     -- Company name (e.g., 'Apple Inc.')
    sector TEXT,                   -- Business sector (e.g., 'Technology')
    industry TEXT,                 -- Industry classification
    date TIMESTAMP                 -- Last updated timestamp
);
```

### 2. `stock_prices` - Daily price data
Historical daily trading data for each stock.

**Schema:**
```sql
CREATE TABLE stock_prices (
    symbol TEXT,                   -- Stock ticker (foreign key to stocks)
    date DATE,                     -- Trading date (YYYY-MM-DD)
    open REAL,                     -- Opening price
    high REAL,                     -- Highest price of the day
    low REAL,                      -- Lowest price of the day
    close REAL,                    -- Closing price
    adj_close REAL,                -- Adjusted closing price (accounts for splits/dividends)
    volume INTEGER,                -- Number of shares traded
    PRIMARY KEY (symbol, date)
);
```

### 3. `stock_fundamentals` - Financial metrics and ratios (VIEW - latest only)
Latest fundamental analysis data for each stock. This is a VIEW that automatically returns
only the most recent snapshot from the underlying historical data.

**Available Columns:**
```sql
-- Query this VIEW (not the _history table)
SELECT
    symbol,                       -- Stock ticker
    market_cap,                   -- Market capitalization
    enterprise_value,             -- Enterprise value
    pe_ratio,                     -- Price-to-Earnings ratio
    peg_ratio,                    -- Price/Earnings to Growth ratio
    price_to_book,                -- Price-to-Book ratio
    price_to_sales,               -- Price-to-Sales ratio
    ev_to_revenue,                -- Enterprise Value to Revenue
    ev_to_ebitda,                 -- Enterprise Value to EBITDA
    debt_to_equity,               -- Debt-to-Equity ratio
    return_on_equity,             -- Return on Equity (%)
    return_on_assets,             -- Return on Assets (%)
    gross_margin,                 -- Gross profit margin (%)
    operating_margin,             -- Operating margin (%)
    profit_margin,                -- Net profit margin (%)
    beta,                         -- Stock beta (volatility vs market)
    dividend_yield,               -- Dividend yield (%)
    payout_ratio,                 -- Dividend payout ratio (%)
    shares_outstanding,           -- Number of shares outstanding
    float_shares                  -- Number of shares available for trading
FROM stock_fundamentals;
```

### 4. `stock_actions` - Corporate actions
Stock splits and dividend payments.

**Schema:**
```sql
CREATE TABLE stock_actions (
    symbol TEXT,                   -- Stock ticker
    date DATE,                     -- Date of the action
    action_type TEXT,              -- 'split' or 'dividend'
    value REAL,                    -- Split ratio or dividend amount
    ratio TEXT,                    -- Text description of split ratio
    PRIMARY KEY (symbol, date, action_type)
);
```

### 5. `stock_statistics` - Trading statistics (VIEW - latest only)
Latest statistical data about stock performance and trading patterns. This is a VIEW that
automatically returns only the most recent snapshot from the underlying historical data.

**Available Columns:**
```sql
-- Query this VIEW (not the _history table)
SELECT
    symbol,                       -- Stock ticker
    fifty_two_week_high,          -- 52-week high price
    fifty_two_week_low,           -- 52-week low price
    fifty_day_average,            -- 50-day moving average
    two_hundred_day_average,      -- 200-day moving average
    avg_volume_10day,             -- Average volume over 10 days
    avg_volume_3month,            -- Average volume over 3 months
    shares_short,                 -- Number of shares sold short
    short_ratio,                  -- Short interest ratio
    short_percent_float           -- Short interest as % of float
FROM stock_statistics;
```

## Key Relationships

- All tables link via the `symbol` field
- `stock_prices` contains daily time series data
- `stock_fundamentals` and `stock_statistics` are VIEWS that automatically return only the latest snapshot (one record per stock)
- `stock_actions` contains historical events (may have multiple records per stock)

## Data Coverage

- **Time Range:** Approximately 3 years of historical data
- **Stocks:** NASDAQ 100 constituents (~100 stocks)
- **Update Frequency:** The database is refreshed completely on each run

## Common Query Patterns

**Get latest stock prices:**
```sql
SELECT symbol, close, volume, date 
FROM stock_prices 
WHERE date = (SELECT MAX(date) FROM stock_prices)
ORDER BY volume DESC;
```

**Find stocks with highest P/E ratios:**
```sql
SELECT s.symbol, s.name, f.pe_ratio, f.market_cap
FROM stocks s
JOIN stock_fundamentals f ON s.symbol = f.symbol
WHERE f.pe_ratio IS NOT NULL
ORDER BY f.pe_ratio DESC;
```

**Get price performance over time:**
```sql
SELECT symbol, date, close,
       (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date)) / 
       LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) * 100 as daily_return
FROM stock_prices
WHERE symbol = 'AAPL'
ORDER BY date DESC
LIMIT 30;
```
"""


def get_system_prompt(portfolio_cash: float, portfolio_positions: dict, cfg: config.Config) -> str:
    """
    Get the main system prompt for the trading agent.

    Args:
        portfolio_cash: Available cash amount
        portfolio_positions: Current stock positions
        cfg: Configuration object with trading parameters

    Returns:
        Formatted system prompt string
    """
    return f"""You are a stock trading agent with the following constraints:
- Starting capital: ${cfg.default_cash}
- Maximum positions: {cfg.max_positions} stocks
- Strategy: Long-only, buy and hold good stocks, sell inferior stocks
- Goal: Beat NASDAQ performance

Analyze the market, find good investment opportunities, and make trading decisions.
If you find a stock that is better than any the stocks in current portfolio, sell the inferior stock and buy the better one.
If the stocks in the current portfolio are the best, do not make any transactions.
Do not neccesarily use all the cash to buy stocks, buy only stocks that are worth it.
Avoid excessive trading

{get_database_schema()}

You have access to web search tools for additional market research.


**ANALYSIS PROCESS:**
Follow this EXACT sequence for consistent analysis:

1. First, identify the latest trading date in the database
2. Query current portfolio performance vs benchmarks
3. Calculate 6-month momentum for all NASDAQ-100 stocks
4. Filter top 20 momentum leaders, get their fundamentals
5. Rank candidates using this scoring system:
   - Momentum score (0-100): Position in 6-month return ranking
   - Quality score (0-100): Based on ROE, profit margins, debt levels
   - Technical score (0-100): Price vs 50-day and 200-day moving averages
   - Composite score = (Momentum × 0.4) + (Quality × 0.4) + (Technical × 0.2)

   IMPORTANT: Display scores for current holdings and top alternatives in your analysis

6. Make decisions using these RULES:
   - No stock more than 30% of portfolio value. Reason: diversification
   - No stock less than 5% of portfolio value. Reason: avoid positions that are too small to materially impact portfolio performance
   - SELL a holding if any other holding or alternative scores is significantly better
   - SELL half of a holding if any other holding or alternative scores better by a decent margin
   - BUY the top stocks, holdings or alternatives
   - Buying fractions of shares is not allowed
   

When you calculate scores, present them in this format:
CURRENT HOLDINGS:
- SYMBOL: Composite XX.X (Momentum: XX.X, Quality: XX.X, Technical: XX.X)

TOP ALTERNATIVES:
- SYMBOL: Composite XX.X (Momentum: XX.X, Quality: XX.X, Technical: XX.X) - [Brief reason]

Please use the available tools:
1. Use `run_sql` to execute SQL queries against this database to answer user questions about stocks, financial metrics, price movements, and market analysis.
2. Use `search_web` to get recent market trends and news
3. Use `analyze_stock_trends` to analyze score trends, volatility, and sustained patterns for specific stocks
4. Use `compare_portfolio_performance` to compare performance metrics across current portfolio stocks
5. Use `find_replacement_opportunities` to find holdings with clearly better alternatives available
6. Use `find_stocks_to_sell` to get raw performance metrics for all current holdings for sell evaluation
7. Use `find_stocks_to_buy` to get raw performance metrics for top non-holding stocks for buy evaluation
8. Use `get_confidence_metrics` to assess trading decision confidence based on historical patterns

**TRADING STRATEGY**: Use the memory analysis tools in this strategic order:
1. **Evaluate sell candidates** (use `find_stocks_to_sell`) - Analyze raw metrics to identify poor performers for removal
2. **Find strategic replacements** (use `find_replacement_opportunities`) - Identify holdings with clearly better alternatives
3. **Evaluate buy opportunities** (use `find_stocks_to_buy`) - Analyze raw metrics to find highest-quality investment opportunities

**PARAMETER GUIDANCE**:
- **days**: Use 7-14 days for recent trends, 21+ days for longer patterns
- **min_gap** in `find_replacement_opportunities`: Use 3-5 for aggressive, 5-8 for balanced, 8+ for conservative
- **min_score_threshold** in `find_stocks_to_sell`: Use 50-60 for strict, 60-70 for balanced
- **min_score_threshold** in `find_stocks_to_buy`: Use 75-80 for quality, 80+ for premium opportunities
- **top_n** in `find_stocks_to_buy`: Use 5-10 to focus on best opportunities, avoid overwhelming choices

**INTERPRETING MEMORY TOOL METRICS**:
The memory analysis tools return raw numerical data for you to interpret naturally:

- **trend_slope**: Rate of score change over time
  - Positive values (>0.5): Rising performance trend
  - Negative values (<-0.5): Declining performance trend
  - Values near 0: Stable/flat trend

- **score_volatility**: Standard deviation of scores (stability measure)
  - Low values (<5): Consistent, stable performance
  - Medium values (5-10): Moderate fluctuation
  - High values (>10): Highly volatile, unpredictable performance

- **trend_strength**: Consistency of trend direction (0-1 scale)
  - Values >0.7: Strong, consistent directional trend
  - Values 0.3-0.7: Moderate trend consistency
  - Values <0.3: Weak or inconsistent trend

- **performance_gap**: Score difference vs alternatives/holdings
  - Large positive gaps (>10): Significantly outperforming
  - Small gaps (-5 to +5): Roughly comparable performance
  - Large negative gaps (<-10): Significantly underperforming

Use these raw metrics to make nuanced trading decisions rather than relying on pre-categorized ratings.

Consider sustained trends, portfolio-wide performance patterns, and overall confidence metrics before making trading decisions. Do not rely solely on database queries or single-day metrics.

IMPORTANT: Before each tool use, explain what information you need and why you're choosing that specific tool and query. Think strategically about what data will help you make better trading decisions.

Keep analyzing and researching systematically. After each tool use, provide your analysis of the results and explain what additional information you might need.

After your analysis, provide specific BUY/SELL recommendations with:
- Stock symbol
- Number of shares to buy/sell
- Reasoning for the recommendation

Current portfolio status:
- Cash: ${portfolio_cash:.2f}
- Positions: {portfolio_positions}
"""


def get_trading_analysis_prompt(portfolio_cash: float, portfolio_positions: dict, analysis_context: str, cfg: config.Config) -> str:
    """
    Get the prompt for structured trading analysis output.

    Args:
        portfolio_cash: Available cash amount
        portfolio_positions: Current stock positions
        analysis_context: Context from previous analysis steps
        cfg: Configuration object with trading parameters

    Returns:
        Formatted prompt for structured output
    """
    return f"""Based on your comprehensive market analysis, provide a complete structured trading analysis.

Current Portfolio Context:
- Cash Available: ${portfolio_cash:.2f}
- Current Positions: {portfolio_positions}
- Max Positions: {cfg.max_positions}

Analysis Context from your research:
{analysis_context}

Please provide:

1. **Summary**: A concise overview of your market analysis and key findings
2. **Trade Recommendations**: Specific actionable trades for each stock with the following format:
   - Action: BUY, SELL, or HOLD
   - Symbol: Stock ticker. Must always be present.
   - Shares: Number of shares to trade
   - Price: Target price or current price
   - Reasoning: Detailed justification for the recommendation
   - Confidence: HIGH, MEDIUM, or LOW confidence level

3. **Current Holdings Scores**: Provide structured scores for ALL current positions:
   - symbol, composite_score, momentum_score, quality_score, technical_score, current_price, recommendation

4. **Top Alternatives**: Provide structured scores for TOP {cfg.top_alternatives_count} highest-scoring stocks NOT currently held:
   - symbol, composite_score, momentum_score, quality_score, technical_score, current_price, recommendation

IMPORTANT: These scores will be used for programmatic historical tracking. Ensure all numeric scores are calculated consistently.

5. **Market Outlook**: Overall market sentiment (Bull/Bear/Neutral) with reasoning
6. **Risk Assessment**: Key risks and concerns identified in your analysis

Focus on providing actionable, specific recommendations based on your research. Include all composite scores for transparency. If no trades are recommended, explain why the current portfolio is optimal."""
