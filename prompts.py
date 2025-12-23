"""
Prompts for the Stock Trading Agent
"""

"""
not part of the latest prompt

Your analysis will proceed in three phases:
1. **Research** - Gather fresh data and calculate current scores
2. **Memory** - Review historical score patterns and trends
3. **Trade** - Make final buy/sell decisions using both contexts
"""


def get_system_prompt(portfolio) -> str:
    """
    Get the main system prompt that introduces the agent.

    Args:
        portfolio: Portfolio object with cash, positions, and configuration

    Returns:
        Formatted system prompt string
    """
    portfolio_summary = portfolio.get_prompt_summary()

    return f"""You are a stock trading agent with the following constraints:
- Starting capital: ${portfolio.cfg.default_cash}
- Maximum positions: {portfolio.cfg.max_positions} stocks
- Strategy: Long-only, buy and hold good stocks, sell inferior stocks
- Goal: Beat NASDAQ-100 performance

Current Portfolio:
- Cash: ${portfolio_summary['cash']:.2f}
- Positions: {portfolio_summary['positions']}
"""


def get_stock_database_schema() -> str:
    """
    Get the stock database schema documentation.

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


def get_research_prompt(portfolio) -> str:
    """
    Get the research prompt for market research and scoring.

    Args:
        portfolio: Portfolio object with prices_as_of date

    Returns:
        Formatted research prompt string
    """
    return f"""# MARKET RESEARCH & SCORING

Your task: Gather fresh market data and calculate scores for holdings and alternatives.

**IMPORTANT:**
- Focus solely on research and scoring
- Do NOT make trading decisions
- All stock data in the database is current as of {portfolio.prices_as_of or "unknown"}

{get_stock_database_schema()}

# YOUR TOOLS

- `run_sql`: Queries the NASDAQ database with 3 years of price history, fundamentals, and statistics
- `search_web`: Retrieves current market news, sentiment, and developments

# WORKFLOW

1. **Gather fresh data** - Use run_sql to query price momentum, fundamentals, technical indicators, and risk metrics
2. **Check market context** - Use search_web for recent news, sentiment, sector trends, and red flags
3. **Calculate scores** - Based on your data analysis, calculate scores for current holdings and promising alternatives

# SCORING METHODOLOGY

Rank candidates using this scoring system:
- Momentum score (0-100): Position in 6-month return ranking
- Quality score (0-100): Based on ROE, profit margins, debt levels
- Technical score (0-100): Price vs 50-day and 200-day moving averages
- Composite score = (Momentum × 0.4) + (Quality × 0.4) + (Technical × 0.2)

Requirements:
- Only score stocks that exist in the NASDAQ-100 database
- Calculate each component score using signals from your SQL queries and news
- All scores must be between 0 and 100

When you have calculated scores for holdings and alternatives, stop calling tools and summarize your findings.
"""


def get_memory_database_schema() -> str:
    """
    Get the memory database schema documentation.

    Returns:
        Database schema documentation string
    """
    return """# Memory Database Schema

**Memory Database Schema:**
```sql
CREATE TABLE agent_scores (
    date DATE,
    symbol TEXT,
    composite_score REAL,
    momentum_score REAL,
    quality_score REAL,
    technical_score REAL,
    current_price REAL,
    is_holding BOOLEAN,  -- 1 = current holding, 0 = alternative
    PRIMARY KEY (date, symbol)
);
```

**Common Query Patterns:**

**Find replacement candidates for a specific holding:**
```sql
-- Compare alternatives vs a specific holding (e.g., AAPL)
SELECT
    a.symbol,
    AVG(a.composite_score) as alternative_avg,
    AVG(h.composite_score) as holding_avg,
    AVG(a.composite_score) - AVG(h.composite_score) as score_gap
FROM agent_scores a
CROSS JOIN agent_scores h
WHERE a.is_holding = 0
  AND h.symbol = 'AAPL'
  AND a.date >= date('now', '-7 days')
  AND h.date >= date('now', '-7 days')
GROUP BY a.symbol
HAVING score_gap > 5
ORDER BY score_gap DESC
LIMIT 10;
```

**Analyze score component breakdown:**
```sql
-- Compare momentum vs quality vs technical scores for a stock
SELECT date,
       momentum_score,
       quality_score,
       technical_score,
       composite_score
FROM agent_scores
WHERE symbol = 'NVDA'
  AND date >= date('now', '-14 days')
ORDER BY date DESC;
```

**Interpreting Analytical Tool Metrics**:
- **trend_slope**: >0.5 = rising, <-0.5 = declining, ~0 = stable
- **score_volatility**: <5 = stable, 5-10 = moderate, >10 = volatile
- **trend_strength**: >0.7 = strong trend, 0.3-0.7 = moderate, <0.3 = weak

**Parameter Guidance**:
- **days**: 7-14 for recent trends, 21+ for longer patterns

Historical patterns reveal momentum shifts invisible in single-day data.
"""


def get_memory_prompt(research_output) -> str:
    """
    Get the prompt for historical analysis.

    Args:
        research_output: Structured output from research with scored stocks

    Returns:
        Formatted prompt for memory analysis
    """
    return f"""# HISTORICAL ANALYSIS

You have completed calculating fresh scores. Now analyze historical score patterns to understand trends.

**IMPORTANT:**
- Do NOT make trading decisions
- Focus solely on identifying trends and patterns in historical data
- Analyze ONLY the stocks scored earlier (listed below)

# TODAY'S FRESH SCORES

{research_output.model_dump_json(indent=2)}

# YOUR TASK

Use memory tools to analyze multi-day patterns for these specific stocks. Focus on:
1. How current holdings' scores have trended over the past 7-14 days
2. Which alternatives show sustained strong scores
3. Stocks with declining score patterns
4. Stocks with rising score patterns

# YOUR TOOLS

**ANALYTICAL TOOLS** (for complex calculations):
- `analyze_stock_trends`: Analyze score trends, volatility, and sustained patterns for specific stocks
  - Returns: trend_slope, score_volatility, trend_strength (R²), avg_score
- `compare_portfolio_performance`: Compare performance metrics across current portfolio stocks
  - Returns: avg_score, trend_slope, score_volatility for each stock

**SQL TOOL** (for flexible queries):
- `run_memory_sql`: Query historical stock scores for custom analysis

{get_memory_database_schema()}

When you have gathered sufficient historical context, stop calling tools and summarize the key patterns you found.
"""


def get_trade_prompt(state) -> str:
    """
    Get the prompt for trading decisions.

    Args:
        state: TradingState with research_output, research_analysis, memory_analysis

    Returns:
        Formatted prompt for structured output
    """
    return f"""# TRADING DECISIONS

You have completed both research (fresh scores) and historical analysis (score patterns). Now make final trading decisions.

Portfolio constraints (cash, positions, max positions) are in your context above.

# DECISION RULES

**Cash Management:**
- Total cost of all BUY recommendations MUST NOT exceed available cash (see context above)
- Consider SELL proceeds first, then calculate remaining cash for BUY orders
- Calculate approximate position sizes: (target allocation × total portfolio value) / current stock price
- Buying fractions of shares is not allowed

**Position Sizing:**
- No stock more than 35% of portfolio value (diversification)
- Avoid stocks less than 5% of portfolio value (immaterial positions)
- Aim for roughly equal-weighted positions when buying multiple stocks

**Trading Logic:**
- SELL a holding if an alternative scores significantly better with sustained strength
- SELL half if an alternative scores moderately better
- BUY top scoring stocks (holdings or alternatives) that show strong patterns
- Process SELL orders first to free up cash for BUY orders

# YOUR ANALYSIS SO FAR

## Fresh Scores from Research

{state["research_output"].model_dump_json(indent=2)}

## Research Analysis

{state["research_analysis"]}

## Memory Analysis

{state["memory_analysis"]}

# PROVIDE STRUCTURED OUTPUT

Based on combining fresh scores with historical patterns, provide:

1. **Summary**: A concise overview of your market analysis and key findings

2. **Trade Recommendations**: Specific actionable trades for each stock with the following format:
   - Action: BUY, SELL, or HOLD
   - Symbol: Stock ticker (required)
   - Shares: Number of shares to trade (required for BUY/SELL)
   - Agent Estimated Price: Current price from research scores (used to calculate total cost)
   - Reasoning: Detailed justification for the recommendation
   - Confidence: HIGH, MEDIUM, or LOW confidence level

3. **Market Outlook**: Overall market sentiment (Bull/Bear/Neutral) with reasoning

4. **Risk Assessment**: Key risks and concerns identified in your analysis

Focus on providing actionable, specific recommendations based on your research. If no trades are recommended, explain why the current portfolio is optimal."""


def get_approval_prompt(state) -> str:
    """
    Get the prompt for approval review.

    Args:
        state: TradingState with all analysis and recommendations

    Returns:
        Formatted prompt for approval review
    """
    return f"""# APPROVAL REVIEW

You are reviewing the proposed trading decisions for final approval.

Review all the analysis and proposed trades for consistency and soundness.

## FRESH SCORES FROM RESEARCH

{state["research_output"].model_dump_json(indent=2)}

## RESEARCH ANALYSIS

{state["research_analysis"]}

## MEMORY ANALYSIS

{state["memory_analysis"]}

## TRADE ANALYSIS

{state["trading_analysis"]}

## PROPOSED TRADING OUTPUT

{state["trading_output"].model_dump_json(indent=2)}

## YOUR TASK

Critically review the proposed trades and make a binary decision: approve or reject all trades.

**Check for these issues:**

1. **Alignment Issues**: Do trades contradict the analysis?
   - Example: "Stock has declining momentum" but recommending BUY
   - Example: Alternative scored higher but not recommending to sell weaker holding

2. **Risk Violations**: Do trades violate portfolio constraints?
   - Any position would exceed 35% of portfolio value
   - Creating positions less than 5% of portfolio value
   - Too many simultaneous changes (over-trading)

3. **Logical Conflicts**: Internal contradictions?
   - Individual assessments conflict with overall market outlook
   - Risk assessment warns about something but trades ignore it

4. **Confidence Issues**: Are recommendations weak or hedged?
   - Multiple LOW confidence trades
   - Reasoning is vague or uncertain

**Decision Guidelines:**
- Set `approved=True` if trades are sound, aligned with analysis, and respect constraints
- Set `approved=False` if ANY significant issue is found (one bad trade rejects all)

Be critical but fair. Minor issues are acceptable if overall strategy is sound.
"""
