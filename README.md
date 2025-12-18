# Stock Trading Agent

LangGraph stock trading agent with the goal to outperform Nasdaq100

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
Automation.py does
- downloads the latest stock data
- updates the portfolio
- runs the agent
- sends an email with the agent analysis and trade recommendations

## Output
The agent provides detailed market analysis and specific buy/sell recommendations with reasoning based on technical indicators, fundamental metrics, and current market conditions.
