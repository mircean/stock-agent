"""
Portfolio class and portfolio value tracking.
Current portfolio state is stored in portfolio.json with the following structure:

{
    "cash": 1000.0,
    "positions": {
        "SYMBOL": {
            "lots": [{"date": "2025-10-09", "shares": 10, "price_per_share": 150.0}],
            "shares": 10,
            "current_price": 155.0,
            "current_value": 1550.0
        }
    },
    "closed_lots": {
        "SYMBOL": [
            {
                "date": "2025-10-09",
                "shares": 5,
                "price_per_share": 150.0,
                "sale_date": "2025-10-15",
                "sale_price": 160.0
            }
        ]
    },
    "total_value": 2550.0,
    "positions_value": 1550.0,
    "prices_as_of": "2025-10-24"
}
"""

import copy
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import config
from portfolio_database import PortfolioDatabase
from stock_history_database import StockHistoryDatabase

logger = logging.getLogger(__name__)


@dataclass
class Lot:
    """
    A lot represents a specific purchase of shares.

    For active lots (in positions):
        - date: purchase date
        - shares: number of shares
        - price_per_share: purchase price per share
        - sale_date: None
        - sale_price: None

    For closed lots (in closed_lots):
        - date: purchase date
        - shares: number of shares sold
        - price_per_share: purchase price per share
        - sale_date: date of sale
        - sale_price: sale price per share
    """

    date: str  # Purchase date in YYYY-MM-DD format
    shares: int  # Number of shares in this lot
    price_per_share: float  # Price paid per share for this lot
    sale_date: str = None  # Sale date in YYYY-MM-DD format (for closed lots)
    sale_price: float = None  # Sale price per share (for closed lots)


class Portfolio:
    """Portfolio management with positions, cash, and lot tracking."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.cash = cfg.default_cash
        self.positions = {}  # dict[symbol] -> {"lots": [Lot], "shares": int, "current_price": float, "current_value": float}
        self.closed_lots = {}  # dict[symbol] -> [Lot]
        self.prices_as_of = None
        self.positions_value = 0.0
        self.total_value = self.cash
        self.nasdaq100 = None  # NASDAQ 100 index value for benchmark tracking

    @classmethod
    def load(cls, cfg: config.Config) -> "Portfolio":
        """Load portfolio from JSON file."""
        with open(cfg.portfolio_file, "r") as f:
            data = json.load(f)

        portfolio = cls(cfg=cfg)

        # Convert lot dictionaries to Lot objects for positions
        positions = data.get("positions", {})
        for symbol, position in positions.items():
            position["lots"] = [Lot(**lot_dict) for lot_dict in position.get("lots", [])]

        # Convert closed_lots dictionaries to Lot objects
        closed_lots = data.get("closed_lots", {})
        for symbol, lots in closed_lots.items():
            closed_lots[symbol] = [Lot(**lot_dict) for lot_dict in lots]

        portfolio.cash = data["cash"]
        portfolio.positions = positions
        portfolio.closed_lots = closed_lots
        portfolio.prices_as_of = data.get("prices_as_of")
        portfolio.positions_value = data["positions_value"]
        portfolio.total_value = data["total_value"]
        portfolio.nasdaq100 = data["nasdaq100"]
        return portfolio

    def save(self):
        """Save portfolio to JSON file and database snapshot."""
        # Convert Lot objects to dictionaries for JSON serialization
        data = {
            "cash": self.cash,
            "positions": {},
            "closed_lots": {},
            "prices_as_of": self.prices_as_of,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "nasdaq100": self.nasdaq100,
        }

        for symbol, position in self.positions.items():
            data["positions"][symbol] = copy.deepcopy(position)
            data["positions"][symbol]["lots"] = [asdict(lot) for lot in position.get("lots", [])]

        for symbol, lots in self.closed_lots.items():
            data["closed_lots"][symbol] = [asdict(lot) for lot in lots]

        # Save to JSON file
        with open(self.cfg.portfolio_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save portfolio snapshot to database for historical tracking
        portfolio_db = PortfolioDatabase(cfg=self.cfg)
        snapshot_date = self.prices_as_of
        portfolio_db.save_portfolio_snapshot(
            date=snapshot_date,
            cash=self.cash,
            positions_value=self.positions_value,
            total_value=self.total_value,
            position_count=len(self.positions),
        )

        # Save NASDAQ 100 benchmark snapshot if available
        if self.nasdaq100 is not None:
            portfolio_db.save_nasdaq100_snapshot(date=snapshot_date, value=self.nasdaq100)

    def to_dict(self) -> dict:
        """Convert portfolio to dictionary for backward compatibility."""
        return {
            "cash": self.cash,
            "positions": self.positions,
            "closed_lots": self.closed_lots,
            "prices_as_of": self.prices_as_of,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "nasdaq100": self.nasdaq100,
        }

    def print(self, label: str = "Portfolio", use_markdown: bool = False) -> str:
        """Build portfolio string in a consistent format."""
        # Header
        if use_markdown:
            text = f"## {label}\n\n"
            text += "| Symbol | Shares | Current Price | Current Value |\n"
            text += "|--------|-------:|:-------------:|--------------:|\n"
        else:
            text = f"📊 {label}:\n"

        # Sort positions by value (descending)
        sorted_positions = sorted(self.positions.items(), key=lambda x: x[1].get("current_value", 0), reverse=True)

        total_positions_value = 0
        for symbol, position in sorted_positions:
            shares = position.get("shares")
            current_price = position.get("current_price", 0)
            value = position.get("current_value", 0)
            total_positions_value += value

            if use_markdown:
                text += f"| {symbol} | {shares:,} | ${current_price:.2f} | ${value:,.2f} |\n"
            else:
                text += f"  {symbol:<4}: \tshares: {shares} \tvalue: ${value:.2f}\n"

        # Footer
        total_value = self.cash + total_positions_value
        if use_markdown:
            text += f"| **Cash** | | | **${self.cash:,.2f}** |\n"
            text += f"| **TOTAL** | | | **${total_value:,.2f}** |\n"
        else:
            text += f"  Cash: ${self.cash:.2f}\n"
            text += f"  Total Value: ${total_value:.2f}"

        return text

    def apply_trades(self, trade_recommendations: list) -> Optional["Portfolio"]:
        """Apply trade recommendations and return new portfolio. Returns None if no trades."""
        logger.info("📈 Applying Recommended Portfolio Changes:")

        # Create database instance for fetching execution prices
        stock_db = StockHistoryDatabase(self.cfg)

        # Create new portfolio with deep copy
        new_portfolio = Portfolio(cfg=self.cfg)
        new_portfolio.cash = self.cash
        new_portfolio.positions = copy.deepcopy(self.positions)
        new_portfolio.closed_lots = copy.deepcopy(self.closed_lots)
        new_portfolio.prices_as_of = self.prices_as_of
        new_portfolio.positions_value = self.positions_value
        new_portfolio.total_value = self.total_value

        trades_recommended = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Sort trades: SELL first, then BUY, then HOLD
        action_priority = {"SELL": 0, "BUY": 1, "HOLD": 2}
        sorted_trades = sorted(trade_recommendations, key=lambda t: action_priority.get(t.action, 3))

        # Process all trades in sorted order
        for trade in sorted_trades:
            symbol = trade.symbol
            assert symbol, "Trade must have symbol"
            assert trade.action in ["SELL", "BUY", "HOLD"], "Trade action must be SELL or BUY or HOLD"
            assert trade.action == "HOLD" or trade.shares, "Trade must have symbol and shares"

            # Get execution price from database for BUY/SELL trades
            execution_price = None
            if trade.action in ["BUY", "SELL"]:
                execution_price = stock_db.get_latest_price(symbol)

            if trade.action == "SELL":
                assert symbol in new_portfolio.positions and "lots" in new_portfolio.positions[symbol], (
                    "Symbol must be in positions and lots must be in positions"
                )
                assert trade.shares <= new_portfolio.positions[symbol]["shares"], "Trade shares must be less than or equal to position shares"

                trades_recommended += 1
                shares_to_sell = int(trade.shares)
                logger.info(f"  {trade.action} {symbol}: {shares_to_sell} shares at ${execution_price:.2f}")

                # Add cash from sale
                new_portfolio.cash += shares_to_sell * execution_price

                current_lots = new_portfolio.positions[symbol]["lots"]

                # Initialize closed_lots for this symbol if needed
                if symbol not in new_portfolio.closed_lots:
                    new_portfolio.closed_lots[symbol] = []

                # Sell from oldest lots first (FIFO)
                remaining_to_sell = shares_to_sell

                while remaining_to_sell > 0:
                    lot = current_lots[0]
                    if lot.shares <= remaining_to_sell:
                        # Sell entire lot - move to closed_lots
                        closed_lot = Lot(date=lot.date, shares=lot.shares, price_per_share=lot.price_per_share, sale_date=today, sale_price=execution_price)
                        new_portfolio.closed_lots[symbol].append(closed_lot)
                        remaining_to_sell -= lot.shares
                        current_lots = current_lots[1:]
                    else:
                        # Partial lot sale
                        shares_sold = remaining_to_sell
                        remaining_shares = lot.shares - remaining_to_sell

                        # Create closed lot for sold portion
                        closed_lot = Lot(date=lot.date, shares=shares_sold, price_per_share=lot.price_per_share, sale_date=today, sale_price=execution_price)
                        new_portfolio.closed_lots[symbol].append(closed_lot)

                        # Update current lot with remaining shares
                        new_lot = Lot(date=lot.date, shares=remaining_shares, price_per_share=lot.price_per_share)
                        current_lots[0] = new_lot
                        remaining_to_sell = 0

                # Update position or remove if no lots remain
                if current_lots:
                    new_portfolio.positions[symbol]["lots"] = current_lots
                    self._update_position(new_portfolio.positions[symbol], execution_price)
                else:
                    del new_portfolio.positions[symbol]

            elif trade.action == "BUY":
                trade_cost = trade.shares * execution_price
                # allow cash to be slighly less than 0
                assert new_portfolio.cash + 100 >= trade_cost, "Cash must be greater than or equal to trade cost"

                trades_recommended += 1
                logger.info(f"  {trade.action} {symbol}: {trade.shares} shares at {execution_price:.2f}")

                new_portfolio.cash -= trade_cost

                # Create new lot
                new_lot = Lot(date=today, shares=int(trade.shares), price_per_share=execution_price)

                if symbol not in new_portfolio.positions:
                    new_portfolio.positions[symbol] = {"lots": []}

                new_portfolio.positions[symbol]["lots"].append(new_lot)
                self._update_position(new_portfolio.positions[symbol], execution_price)

        if trades_recommended > 0:
            # Recalculate portfolio totals
            new_portfolio.positions_value = sum(pos["current_value"] for pos in new_portfolio.positions.values())
            new_portfolio.total_value = new_portfolio.cash + new_portfolio.positions_value

            logger.info(f"✅ {trades_recommended} trades recommended and simulated")
            return new_portfolio

        logger.info("ℹ️ No valid trades to execute")
        return None

    @staticmethod
    def _update_position(position: dict, current_price: float):
        """Update position with current price and value."""
        position["shares"] = sum(lot.shares for lot in position["lots"])
        position["current_price"] = current_price
        position["current_value"] = position["shares"] * position["current_price"]

    def update_stock_prices(self):
        """Update portfolio values based on current stock prices and save portfolio and benchmark snapshots."""
        stock_history_db_path = self.cfg.stock_history_db_name

        # Get the most recent date in the stock_prices table
        with sqlite3.connect(stock_history_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MAX(date) FROM stock_prices
                """
            )
            result = cursor.fetchone()

        assert result and result[0], "No stock price data found in database"
        latest_price_date = result[0]

        logger.info("=" * 50)
        logger.info("UPDATING PORTFOLIO VALUES")
        logger.info("=" * 50)
        logger.info(f"Using prices as of: {latest_price_date}")

        total_positions_value = 0

        for symbol, position in self.positions.items():
            shares = position.get("shares", 0)

            # Get latest price from market database
            with sqlite3.connect(stock_history_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT close, date FROM stock_prices
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                    """,
                    (symbol,),
                )
                result = cursor.fetchall()

            assert result, f"No price data found for {symbol}"

            current_price = float(result[0][0])
            price_date = result[0][1]
            current_value = shares * current_price
            total_positions_value += current_value

            # Update position with current price and value
            position["current_price"] = current_price
            position["current_value"] = current_value

            logger.info(f"{symbol}: {shares} shares @ ${current_price:.2f} = ${current_value:.2f} [{price_date}]")

        total_portfolio_value = self.cash + total_positions_value

        # Update portfolio totals
        self.positions_value = total_positions_value
        self.total_value = total_portfolio_value
        self.prices_as_of = latest_price_date

        # Fetch NASDAQ 100 benchmark value for the same date
        with sqlite3.connect(stock_history_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE symbol = '^NDX'
                ORDER BY date DESC
                LIMIT 1
                """
            )
            ndx_result = cursor.fetchone()

        assert ndx_result, "NASDAQ 100 (^NDX) data not found in database"
        self.nasdaq100 = float(ndx_result[0])

        # Save portfolio and benchmark (handles JSON + DB snapshots)
        self.save()

        logger.info("PORTFOLIO SUMMARY:")
        logger.info(f"Cash: ${self.cash:.2f}")
        logger.info(f"Positions value: ${total_positions_value:.2f}")
        logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")
        logger.info(f"NASDAQ 100: {self.nasdaq100:.2f}")
        logger.info(f"Portfolio updated and snapshot saved for {latest_price_date}")
        logger.info("=" * 50)
