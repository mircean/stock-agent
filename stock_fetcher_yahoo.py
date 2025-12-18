import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config
from stock_history_database import StockHistoryDatabase

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)


class StockFetcher:
    """Comprehensive stock fetcher for NASDAQ 100"""

    def __init__(self, db):
        self.db = db
        self.rate_limit_delay = 0.1  # Delay between requests to avoid rate limiting

        # Date range for 3 years (use UTC for Yahoo Finance API)
        # Note: yfinance end date is exclusive, so add 1 day to include today
        self.end_date = datetime.now(timezone.utc) + timedelta(days=1)
        self.start_date = self.end_date - timedelta(days=3 * 365)

    def fetch_stock_info(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive stock information using yfinance"""

        ticker = yf.Ticker(symbol)

        # Get stock info
        info = ticker.info
        assert info is not None, f"No data available for {symbol}"
        assert info.get("regularMarketPrice") is not None, f"No data available for {symbol}"

        # Get historical data
        history = ticker.history(
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        assert not history.empty, f"No historical data available for {symbol}"

        latest_date = history.index[-1].to_pydatetime()

        # If market is open (REGULAR), add today's intraday data
        # We reuse the "Close" column for the live price even though the day hasn't closed yet
        # This allows existing queries to work without schema changes
        if info.get("marketState") == "REGULAR":
            latest_date = pd.Timestamp(datetime.now(timezone.utc).date())

            # Create today's row with live price in Close column
            new_row = pd.DataFrame(
                {
                    "Open": [float(info["regularMarketOpen"])],
                    "High": [float(info["regularMarketDayHigh"])],
                    "Low": [float(info["regularMarketDayLow"])],
                    "Close": [float(info["regularMarketPrice"])],  # Live price (day hasn't closed yet)
                    "Adj Close": [float(0.0)],  # Cannot calculate until day closes
                    "Volume": [float(0.0)],  # Volume incomplete during trading
                },
                index=[latest_date],
            )
            history = pd.concat([history, new_row])
            logger.debug(f"Added intraday data for {symbol}: Close=${float(info['regularMarketPrice']):.2f} (live price, market open)")

        # Get actions (splits and dividends)
        actions = ticker.actions
        if actions.empty:
            # Create empty DataFrame with proper structure
            actions = pd.DataFrame(columns=["Dividends", "Stock Splits"])

        return {
            "symbol": symbol,
            "info": info,
            "history": history,
            "actions": actions,
            "latest_date": latest_date,
        }

    def process_stock_data(self, stock_data: Dict) -> bool:
        """Process and store stock data in database"""
        symbol = stock_data["symbol"]
        info = stock_data["info"]

        # Insert stock metadata
        self.db.insert_stock(
            symbol=symbol,
            name=info.get("longName", info.get("shortName", symbol)),
            sector=info.get("sector"),
            industry=info.get("industry"),
        )

        # Round and insert price data
        if not stock_data["history"].empty:
            history = stock_data["history"].copy()
            # Round all price columns to 2 decimal places
            price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
            for col in price_columns:
                if col in history.columns:
                    history[col] = history[col].round(2)
            self.db.insert_price_data(symbol, history)

        # Round and insert actions (splits and dividends)
        if not stock_data["actions"].empty:
            actions = stock_data["actions"].copy()
            # Round dividend prices to 2 decimal places
            if "Dividends" in actions.columns:
                actions["Dividends"] = actions["Dividends"].round(2)
            self.db.insert_stock_actions(symbol, actions)

        # Prepare fundamentals data
        fundamentals = {
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "trailingPE": info.get("trailingPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "debtToEquity": info.get("debtToEquity"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "profitMargins": info.get("profitMargins"),
            "beta": info.get("beta"),
            "dividendYield": info.get("dividendYield"),
            "payoutRatio": info.get("payoutRatio"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "floatShares": info.get("floatShares"),
        }

        # Clean up None values, round floats to 2 decimals
        cleaned_fundamentals = {}
        for key, value in fundamentals.items():
            if value is None:
                continue
            if isinstance(value, float):
                if not np.isnan(value):
                    cleaned_fundamentals[key] = round(value, 2)
            else:
                # Keep other types as-is (int, etc.)
                assert isinstance(value, int) or isinstance(value, str), f"Unexpected type: {type(value)}"
                cleaned_fundamentals[key] = value

        if cleaned_fundamentals:
            self.db.insert_fundamentals(symbol, cleaned_fundamentals, stock_data["latest_date"])

        # Prepare statistics data
        statistics = {
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            "averageVolume10days": info.get("averageVolume10days"),
            "averageVolume3Month": info.get("averageVolume3Month"),
            "sharesShort": info.get("sharesShort"),
            "shortRatio": info.get("shortRatio"),
            "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        }

        # Clean up statistics data, round floats to 2 decimals
        cleaned_statistics = {}
        for key, value in statistics.items():
            if value is None:
                continue
            if isinstance(value, float):
                if not np.isnan(value):
                    cleaned_statistics[key] = round(value, 2)
            else:
                # Keep other types as-is (int, etc.)
                assert isinstance(value, int) or isinstance(value, str), f"Unexpected type: {type(value)}"
                cleaned_statistics[key] = value

        if cleaned_statistics:
            self.db.insert_statistics(symbol, cleaned_statistics, stock_data["latest_date"])

    def fetch_single_stock(self, symbol: str) -> bool:
        """Fetch and process data for a single stock"""
        logger.info(f"Fetching data for {symbol}")
        for _ in range(3):
            # Add delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            try:
                stock_data = self.fetch_stock_info(symbol)
                break
            # except yf.YFRateLimitError as e:
            #     logger.error(f"Rate limit exceeded for {symbol}: {e}")
            #     self.rate_limit_delay *= 2
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        assert stock_data, f"Failed to fetch data for {symbol}"
        try:
            self.process_stock_data(stock_data)
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            raise e

    def fetch_stocks(
        self,
        symbols: List[str],
    ):
        """Fetch data for NASDAQ 100 stocks sequentially"""
        logger.info(f"Processing {len(symbols)} stocks...")

        # Process stocks sequentially with progress bar
        for symbol in tqdm(symbols, desc="Fetching stock data"):
            self.fetch_single_stock(symbol)

        # Print database statistics
        stats = self.db.get_database_stats()
        logger.info(f"Database stats: {stats}")

    def update_single_stock(self, symbol: str) -> bool:
        """Update data for a single stock"""
        logger.info(f"Updating {symbol}...")
        return self.fetch_single_stock(symbol)

    def get_stock_summary(self, symbol: str) -> Optional[Dict]:
        """Get a summary of stock data"""
        data = self.db.get_stock_data(symbol)
        if data["prices"].empty:
            return None

        latest_price = data["prices"].iloc[-1]
        price_range = data["prices"]["close"]

        summary = {
            "symbol": symbol,
            "latest_date": latest_price["date"],
            "latest_close": latest_price["close"],
            "volume": latest_price["volume"],
            "52_week_high": price_range.max(),
            "52_week_low": price_range.min(),
            "price_change_1y": ((latest_price["close"] - data["prices"].iloc[0]["close"]) / data["prices"].iloc[0]["close"]) * 100,
            "data_points": len(data["prices"]),
        }

        if not data["fundamentals"].empty:
            latest_fundamentals = data["fundamentals"].iloc[0]
            summary.update(
                {
                    "pe_ratio": latest_fundamentals.get("pe_ratio"),
                    "market_cap": latest_fundamentals.get("market_cap"),
                    "dividend_yield": latest_fundamentals.get("dividend_yield"),
                }
            )

        return summary


if __name__ == "__main__":
    cfg = config.parse_config()
    cfg.stock_history_db_name = "stock_history_test.db"

    # Example usage
    db = StockHistoryDatabase(cfg)
    fetcher = StockFetcher(db)

    # Fetch data for a few major stocks as test
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NDX"]

    print("Testing with major stocks...")
    fetcher.fetch_stocks(symbols=test_symbols)

    # Print summaries
    for symbol in test_symbols:
        summary = fetcher.get_stock_summary(symbol)
        if summary:
            print(f"\n{symbol}: ${summary['latest_close']:.2f}, Volume: {summary['volume']:,}, Data points: {summary['data_points']}")

    # Clean up test database
    os.remove("stock_history_test.db")
    print("\n✅ Test database deleted")
