import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

import config
import pandas as pd

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)


class StockHistoryDatabase:
    """Database manager for stock data storage and retrieval"""

    def __init__(self, cfg: config.Config):
        self.db_path = cfg.stock_history_db_name
        self.init_database()

    def init_database(self):
        """Initialize database with required tables from schema file"""
        schema_path = os.path.join(os.path.dirname(__file__), "stock_history_schema.sql")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(schema_sql)
            conn.commit()
            # logger.info("Database initialized successfully from stock_history_schema.sql")

    def insert_stock(self, symbol: str, name: str, sector: str = None, industry: str = None):
        """Insert or update stock metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO stocks (symbol, name, sector, industry, date)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (symbol, name, sector, industry),
            )
            conn.commit()

    def insert_price_data(self, symbol: str, price_data: pd.DataFrame):
        """Insert historical price data for a stock"""
        with sqlite3.connect(self.db_path) as conn:
            for date, row in price_data.iterrows():
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO stock_prices 
                    (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        date.strftime("%Y-%m-%d"),
                        float(row.get("Open", 0)) if pd.notna(row.get("Open")) else None,
                        float(row.get("High", 0)) if pd.notna(row.get("High")) else None,
                        float(row.get("Low", 0)) if pd.notna(row.get("Low")) else None,
                        float(row.get("Close", 0)) if pd.notna(row.get("Close")) else None,
                        float(row.get("Adj Close", 0)) if pd.notna(row.get("Adj Close")) else None,
                        int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
                    ),
                )
            conn.commit()

    def insert_fundamentals(self, symbol: str, fundamentals: Dict):
        """Insert fundamental data for a stock (writes to history table, view shows latest)"""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert into history table (stock_fundamentals view will show latest automatically)
            cursor.execute(
                """
                INSERT OR REPLACE INTO stock_fundamentals_history
                (symbol, date, market_cap, enterprise_value, pe_ratio, peg_ratio, price_to_book,
                 price_to_sales, ev_to_revenue, ev_to_ebitda, debt_to_equity, return_on_equity,
                 return_on_assets, gross_margin, operating_margin, profit_margin, beta,
                 dividend_yield, payout_ratio, shares_outstanding, float_shares)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    date,
                    fundamentals.get("marketCap"),
                    fundamentals.get("enterpriseValue"),
                    fundamentals.get("trailingPE"),
                    fundamentals.get("pegRatio"),
                    fundamentals.get("priceToBook"),
                    fundamentals.get("priceToSalesTrailing12Months"),
                    fundamentals.get("enterpriseToRevenue"),
                    fundamentals.get("enterpriseToEbitda"),
                    fundamentals.get("debtToEquity"),
                    fundamentals.get("returnOnEquity"),
                    fundamentals.get("returnOnAssets"),
                    fundamentals.get("grossMargins"),
                    fundamentals.get("operatingMargins"),
                    fundamentals.get("profitMargins"),
                    fundamentals.get("beta"),
                    fundamentals.get("dividendYield"),
                    fundamentals.get("payoutRatio"),
                    fundamentals.get("sharesOutstanding"),
                    fundamentals.get("floatShares"),
                ),
            )
            conn.commit()

    def insert_stock_actions(self, symbol: str, actions: pd.DataFrame):
        """Insert stock splits and dividends"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Handle dividends
            if "Dividends" in actions.columns:
                for date, row in actions.iterrows():
                    dividend = row.get("Dividends", 0)
                    if dividend > 0:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO stock_actions
                            (symbol, date, action_type, value)
                            VALUES (?, ?, 'dividend', ?)
                        """,
                            (symbol, date.strftime("%Y-%m-%d"), float(dividend)),
                        )

            # Handle stock splits
            if "Stock Splits" in actions.columns:
                for date, row in actions.iterrows():
                    split = row.get("Stock Splits", 0)
                    if split > 0:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO stock_actions
                            (symbol, date, action_type, value, ratio)
                            VALUES (?, ?, 'split', ?, ?)
                        """,
                            (
                                symbol,
                                date.strftime("%Y-%m-%d"),
                                float(split),
                                str(split),
                            ),
                        )

            conn.commit()

    def insert_statistics(self, symbol: str, stats: Dict):
        """Insert stock statistics (writes to history table, view shows latest)"""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert into history table (stock_statistics view will show latest automatically)
            cursor.execute(
                """
                INSERT OR REPLACE INTO stock_statistics_history
                (symbol, date, fifty_two_week_high, fifty_two_week_low, fifty_day_average,
                 two_hundred_day_average, avg_volume_10day, avg_volume_3month, shares_short,
                 short_ratio, short_percent_float)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    date,
                    stats.get("fiftyTwoWeekHigh"),
                    stats.get("fiftyTwoWeekLow"),
                    stats.get("fiftyDayAverage"),
                    stats.get("twoHundredDayAverage"),
                    stats.get("averageVolume10days"),
                    stats.get("averageVolume3Month"),
                    stats.get("sharesShort"),
                    stats.get("shortRatio"),
                    stats.get("shortPercentOfFloat"),
                ),
            )
            conn.commit()

    def get_existing_stocks(self) -> List[str]:
        """Get list of stocks already in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM stocks")
            return [row[0] for row in cursor.fetchall()]

    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """Retrieve comprehensive stock data"""
        with sqlite3.connect(self.db_path) as conn:
            # Base query conditions
            date_condition = ""
            params = [symbol]

            if start_date:
                date_condition += " AND date >= ?"
                params.append(start_date)
            if end_date:
                date_condition += " AND date <= ?"
                params.append(end_date)

            # Get price data
            price_query = f"""
                SELECT date, open, high, low, close, adj_close, volume
                FROM stock_prices 
                WHERE symbol = ? {date_condition}
                ORDER BY date
            """
            price_df = pd.read_sql_query(price_query, conn, params=params)

            # Get fundamentals
            fund_query = f"""
                SELECT * FROM stock_fundamentals 
                WHERE symbol = ? {date_condition}
            """
            fundamentals_df = pd.read_sql_query(fund_query, conn, params=params)

            # Get actions
            actions_query = f"""
                SELECT date, action_type, value, ratio FROM stock_actions 
                WHERE symbol = ? {date_condition}
                ORDER BY date
            """
            actions_df = pd.read_sql_query(actions_query, conn, params=params)

            # Get statistics
            stats_query = f"""
                SELECT * FROM stock_statistics 
                WHERE symbol = ? {date_condition}
            """
            stats_df = pd.read_sql_query(stats_query, conn, params=params)

            return {
                "prices": price_df,
                "fundamentals": fundamentals_df,
                "actions": actions_df,
                "statistics": stats_df,
            }

    def cleanup_old_data(self, years: int = 3):
        """
        Remove data older than specified years from stock_prices and table.
        No cleanup for stock_actions table - this is always from the begining of time

        Args:
            years: Number of years of history to keep (default: 3)
        """
        from datetime import datetime, timedelta, timezone

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clean up old price data
            cursor.execute("DELETE FROM stock_prices WHERE date < ?", (cutoff_date,))
            deleted_prices = cursor.rowcount
            logger.info(f"Deleted {deleted_prices} old price records before {cutoff_date}")

            conn.commit()

            # no vacuum, database is small enough

    def remove_data_after_date(self, date: str):
        """
        Remove all data newer than specified date (useful for backtesting setup).

        Args:
            date: Date in YYYY-MM-DD format. All data after this date will be deleted.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove price data after date
            cursor.execute("DELETE FROM stock_prices WHERE date > ?", (date,))
            deleted_prices = cursor.rowcount

            # Remove fundamentals history after date
            cursor.execute("DELETE FROM stock_fundamentals_history WHERE date > ?", (date,))
            deleted_fundamentals = cursor.rowcount

            # Remove statistics history after date
            cursor.execute("DELETE FROM stock_statistics_history WHERE date > ?", (date,))
            deleted_statistics = cursor.rowcount

            # Remove actions after date
            cursor.execute("DELETE FROM stock_actions WHERE date > ?", (date,))
            deleted_actions = cursor.rowcount

            conn.commit()

            logger.info(f"Removed data after {date}:")
            logger.info(f"  - {deleted_prices} price records")
            logger.info(f"  - {deleted_fundamentals} fundamental records")
            logger.info(f"  - {deleted_statistics} statistics records")
            logger.info(f"  - {deleted_actions} action records")

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            stats = {}

            # Count stocks
            cursor.execute("SELECT COUNT(*) FROM stocks")
            stats["total_stocks"] = cursor.fetchone()[0]

            # Count price records
            cursor.execute("SELECT COUNT(*) FROM stock_prices")
            stats["price_records"] = cursor.fetchone()[0]

            # Date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices")
            date_range = cursor.fetchone()
            stats["date_range"] = {"start": date_range[0], "end": date_range[1]}

            # Database size
            stats["db_size_mb"] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)

            return stats


# MemoryDatabase moved to memory_database.py


if __name__ == "__main__":
    db = StockHistoryDatabase()
    print("Database initialized")
    print("Stats:", db.get_database_stats())
