"""
NASDAQ Stock Database Builder

Downloads NASDAQ 100 stock data and stores it in a SQLite database.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List

import config
import requests
from stock_fetcher_yahoo import StockFetcher
from stock_history_database import StockHistoryDatabase

logger = logging.getLogger(__name__)


class NasdaqStockFetcher:
    """Fetches list of NASDAQ-listed stocks"""

    def __init__(self):
        self.base_url = "https://www.nasdaq.com/market-activity/stocks/screener"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def get_nasdaq_stocks(self) -> List[Dict[str, str]]:
        """
        Fetch list of NASDAQ stocks
        Returns list of dictionaries with symbol, name, and other basic info
        """
        # Use the official NASDAQ 100 API endpoint
        url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"

        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        stocks = []

        rows = data["data"]["data"]["rows"]
        for row in rows:
            symbol = row.get("symbol", "").strip()
            assert symbol, f"No symbol found for {row}"
            # ignore GOOGL, same as GOOG
            if symbol == "GOOGL":
                continue

            stocks.append(
                {
                    "symbol": symbol,
                    "name": row.get("companyName", row.get("name", "")).strip(),
                    "market_cap": "",  # NASDAQ 100 API may not include market cap
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                }
            )

        logger.info(f"Fetched {len(stocks)} NASDAQ 100 stocks")
        return stocks


def main():
    """Main application entry point"""
    cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    # Reduce yfinance logging noise
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("NASDAQ Stock Database Builder")
    logger.info(f"Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize components
    nasdaq_fetcher = NasdaqStockFetcher()
    db = StockHistoryDatabase(cfg)
    fetcher = StockFetcher(db)

    # Fetch NASDAQ 100 stock list and include ^NDX (NASDAQ 100 Index)
    nasdaq_stocks = nasdaq_fetcher.get_nasdaq_stocks()
    symbols = ["^NDX"] + [stock["symbol"] for stock in nasdaq_stocks if stock["symbol"]]

    # Download and store NASDAQ 100 data
    logger.info("Downloading NASDAQ 100 stock data...")
    fetcher.fetch_stocks(symbols)

    # Cleanup old data (keep last 3 years)
    logger.info("Cleaning up data older than 3 years...")
    db.cleanup_old_data(years=3)

    # Show final stats
    stats = db.get_database_stats()
    logger.info("=" * 50)
    logger.info("DATABASE STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total stocks: {stats['total_stocks']:,}")
    logger.info(f"Price records: {stats['price_records']:,}")
    logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    logger.info(f"Database size: {stats['db_size_mb']} MB")
    logger.info("=" * 50)

    logger.info(f"Completed at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
