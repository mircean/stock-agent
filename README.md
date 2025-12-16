# Stock Trading Agent

AI-powered stock trading agent with the goal to outperform Nasdaq100

## Performance vs Nasday 100

<img width="1400" height="866" alt="image" src="https://github.com/user-attachments/assets/49dd26f2-02f4-48ec-b6fd-d8e1bbd1956f" />

## Simulations 

<img width="1399" height="868" alt="image" src="https://github.com/user-attachments/assets/762927be-0da6-48c1-bce9-420296229403" />

## Setup

1. Install dependencies: `uv sync`
2. Set environment variables in `.env`:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `LANGSMITH_API_KEY`

## Scripts
Run `automation.py` to download the data and run the agent

## Features

- **Historical Analysis**: 3 years of NASDAQ 100 stock data (OHLCV, fundamentals, statistics)
- **Market Intelligence**: Real-time news integration via web search
- **Portfolio Management**: Track positions, cash, and performance ($10000 starting capital)
- **Trading Constraints**: Long-only strategy, max 10 positions
- **Data Storage**: Local SQLite, Postgresql databases for storing stock prices, portfolio, memory
- **Sends email with the trading analysis**

## Output

The agent provides detailed market analysis and specific buy/sell recommendations with reasoning based on technical indicators, fundamental metrics, and current market conditions.
