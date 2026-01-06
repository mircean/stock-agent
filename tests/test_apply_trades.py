#!/usr/bin/env python3
"""
Unit tests for apply_trades_to_portfolio function - Basic functionality tests
"""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agent import TradeRecommendation
from portfolio import Lot, Portfolio


class TestApplyTradesToPortfolioBasic(unittest.TestCase):
    """Basic functionality tests for apply_trades_to_portfolio function"""

    def setUp(self):
        """Set up test fixtures"""
        cfg = config.Config()

        # Mock prices for testing
        self.mock_prices = {
            "NVDA": 100.0,
            "AAPL": 150.0,
            "GOOG": 510.0,
            "MSFT": 130.0,
        }

        # Create a side effect function for the mock
        def mock_get_price(symbol):
            return self.mock_prices.get(symbol, 100.0)

        self.mock_get_price = mock_get_price

        # Create initial portfolio
        self.initial_portfolio = Portfolio(cfg)
        self.initial_portfolio.cash = 1000.0
        self.initial_portfolio.positions = {
            "AAPL": {"lots": [Lot(date="2025-10-01", shares=10, price_per_share=150.0)], "shares": 10, "current_price": 150.0, "current_value": 1500.0},
            "GOOG": {"lots": [Lot(date="2025-10-02", shares=5, price_per_share=500.0)], "shares": 5, "current_price": 500.0, "current_value": 2500.0},
        }
        self.initial_portfolio.closed_lots = {}

        # Portfolio with multiple lots for FIFO testing
        self.multi_lot_portfolio = Portfolio(cfg)
        self.multi_lot_portfolio.cash = 2000.0
        self.multi_lot_portfolio.positions = {
            "MSFT": {
                "lots": [
                    Lot(date="2025-09-01", shares=10, price_per_share=100.0),  # Oldest
                    Lot(date="2025-09-15", shares=20, price_per_share=110.0),  # Middle
                    Lot(date="2025-10-01", shares=15, price_per_share=120.0),  # Newest
                ],
                "shares": 45,  # 10 + 20 + 15
                "current_price": 120.0,
                "current_value": 5400.0,  # 45 * 120
            }
        }
        self.multi_lot_portfolio.closed_lots = {}

    def test_1_no_trades(self):
        """Test 1: Empty trade recommendations should return None"""
        result = self.initial_portfolio.apply_trades([])

        # Should return None when no trades
        self.assertIsNone(result)

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_2_buy_new_position(self, mock_get_latest_price):
        """Test 2: Buy new position should add a new lot to portfolio"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="BUY", symbol="NVDA", shares=10, agent_estimated_price=100.0, reasoning="New position")]

        result = self.initial_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash is reduced by purchase amount (10 shares * $100)
        expected_cash = 1000.0 - (10 * 100.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify new position is added with correct values
        self.assertIn("NVDA", result.positions)
        self.assertEqual(result.positions["NVDA"]["shares"], 10)
        self.assertEqual(result.positions["NVDA"]["current_price"], 100.0)
        self.assertEqual(result.positions["NVDA"]["current_value"], 1000.0)

        # Verify new lot was created
        self.assertEqual(len(result.positions["NVDA"]["lots"]), 1)
        nvda_lot = result.positions["NVDA"]["lots"][0]
        self.assertEqual(nvda_lot.shares, 10)
        self.assertEqual(nvda_lot.price_per_share, 100.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(nvda_lot.date, today)  # Should be today's date

        # Verify original positions remain unchanged
        self.assertEqual(result.positions["AAPL"]["shares"], 10)
        self.assertEqual(result.positions["GOOG"]["shares"], 5)

        # Verify total number of positions increased by 1
        self.assertEqual(len(result.positions), 3)  # AAPL, GOOG, NVDA

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_3_buy_existing_position(self, mock_get_latest_price):
        """Test 3: Buy existing position should add shares to existing lot"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="BUY", symbol="AAPL", shares=5, agent_estimated_price=150.0, reasoning="Add to position")]

        result = self.initial_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash is reduced by purchase amount (5 shares * $150)
        expected_cash = 1000.0 - (5 * 150.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify existing position is increased
        self.assertEqual(result.positions["AAPL"]["shares"], 15)  # 10 + 5
        self.assertEqual(result.positions["AAPL"]["current_price"], 150.0)
        self.assertEqual(result.positions["AAPL"]["current_value"], 2250.0)  # 15 * 150

        # Verify new lot was added to existing position
        self.assertEqual(len(result.positions["AAPL"]["lots"]), 2)  # Original + new
        # Check the new lot
        new_lot = result.positions["AAPL"]["lots"][1]  # Second lot
        self.assertEqual(new_lot.shares, 5)
        self.assertEqual(new_lot.price_per_share, 150.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(new_lot.date, today)  # Should be today's date

        # Verify other positions remain unchanged
        self.assertEqual(result.positions["GOOG"]["shares"], 5)

        # Verify total number of positions remains the same
        self.assertEqual(len(result.positions), 2)  # Still AAPL, GOOG

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_4_sell_entire_position_single_lot(self, mock_get_latest_price):
        """Test 4: Sell entire position with single lot"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="SELL", symbol="AAPL", shares=10, agent_estimated_price=150.0, reasoning="Sell all AAPL")]

        result = self.initial_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash increased by sale proceeds (10 shares * $150)
        expected_cash = 1000.0 + (10 * 150.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify AAPL position is completely removed
        self.assertNotIn("AAPL", result.positions)

        # Verify closed_lots contains the sold lot
        self.assertIsNotNone(result.closed_lots)
        self.assertIn("AAPL", result.closed_lots)
        self.assertEqual(len(result.closed_lots["AAPL"]), 1)
        closed_lot = result.closed_lots["AAPL"][0]
        self.assertEqual(closed_lot.shares, 10)
        self.assertEqual(closed_lot.price_per_share, 150.0)  # Original purchase price
        self.assertEqual(closed_lot.sale_price, 150.0)  # Sale price from mock
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(closed_lot.sale_date, today)  # Today's date
        self.assertEqual(closed_lot.date, "2025-10-01")  # Original purchase date

        # Verify GOOG position remains unchanged
        self.assertEqual(result.positions["GOOG"]["shares"], 5)
        self.assertEqual(len(result.positions["GOOG"]["lots"]), 1)

        # Verify total number of positions decreased by 1
        self.assertEqual(len(result.positions), 1)  # Only GOOG

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_5_sell_partial_position_single_lot(self, mock_get_latest_price):
        """Test 5: Sell partial position from single lot"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="SELL", symbol="AAPL", shares=6, agent_estimated_price=150.0, reasoning="Partial AAPL sale")]

        result = self.initial_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash increased by sale proceeds (6 shares * $150)
        expected_cash = 1000.0 + (6 * 150.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify AAPL position is reduced
        self.assertIn("AAPL", result.positions)
        self.assertEqual(result.positions["AAPL"]["shares"], 4)  # 10 - 6 = 4
        self.assertEqual(result.positions["AAPL"]["current_price"], 150.0)
        self.assertEqual(result.positions["AAPL"]["current_value"], 600.0)  # 4 * 150

        # Verify remaining lot has correct shares
        self.assertEqual(len(result.positions["AAPL"]["lots"]), 1)
        remaining_lot = result.positions["AAPL"]["lots"][0]
        self.assertEqual(remaining_lot.shares, 4)
        self.assertEqual(remaining_lot.price_per_share, 150.0)  # Original purchase price
        self.assertEqual(remaining_lot.date, "2025-10-01")  # Original date

        # Verify closed_lots contains the partial sale
        self.assertIsNotNone(result.closed_lots)
        self.assertIn("AAPL", result.closed_lots)
        self.assertEqual(len(result.closed_lots["AAPL"]), 1)
        closed_lot = result.closed_lots["AAPL"][0]
        self.assertEqual(closed_lot.shares, 6)  # Sold portion
        self.assertEqual(closed_lot.price_per_share, 150.0)  # Original purchase price
        self.assertEqual(closed_lot.sale_price, 150.0)  # Sale price from mock
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(closed_lot.sale_date, today)  # Today's date
        self.assertEqual(closed_lot.date, "2025-10-01")  # Original purchase date

        # Verify GOOG position remains unchanged
        self.assertEqual(result.positions["GOOG"]["shares"], 5)
        self.assertEqual(len(result.positions), 2)  # Both positions remain

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_6_sell_fifo_multiple_lots_complete_oldest(self, mock_get_latest_price):
        """Test 6: FIFO sell that completely uses oldest lot"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="SELL", symbol="MSFT", shares=10, agent_estimated_price=200.0, reasoning="Sell 10 MSFT shares")]

        result = self.multi_lot_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash increased by sale proceeds (10 shares * $130)
        expected_cash = 2000.0 + (10 * 130.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify MSFT position is reduced
        self.assertEqual(result.positions["MSFT"]["shares"], 35)  # 45 - 10 = 35
        self.assertEqual(result.positions["MSFT"]["current_price"], 130.0)
        self.assertEqual(result.positions["MSFT"]["current_value"], 4550.0)  # 35 * 130

        # Verify oldest lot (2025-09-01) is completely removed
        self.assertEqual(len(result.positions["MSFT"]["lots"]), 2)  # Only 2 lots remain

        # Verify remaining lots are the middle and newest ones
        remaining_lots = result.positions["MSFT"]["lots"]
        self.assertEqual(remaining_lots[0].date, "2025-09-15")  # Middle lot now first
        self.assertEqual(remaining_lots[0].shares, 20)
        self.assertEqual(remaining_lots[0].price_per_share, 110.0)

        self.assertEqual(remaining_lots[1].date, "2025-10-01")  # Newest lot
        self.assertEqual(remaining_lots[1].shares, 15)
        self.assertEqual(remaining_lots[1].price_per_share, 120.0)

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_7_sell_fifo_multiple_lots_partial_second(self, mock_get_latest_price):
        """Test 7: FIFO sell that uses complete oldest lot and partial second lot"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="SELL", symbol="MSFT", shares=25, agent_estimated_price=200.0, reasoning="Sell 25 MSFT shares")]

        result = self.multi_lot_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash increased by sale proceeds
        self.assertGreater(result.cash, 2000.0)

        # Verify MSFT position is reduced
        self.assertEqual(result.positions["MSFT"]["shares"], 20)  # 45 - 25 = 20
        self.assertIsNotNone(result.positions["MSFT"]["current_price"])
        self.assertGreater(result.positions["MSFT"]["current_value"], 0)

        # Verify lots: oldest completely removed, middle partially consumed, newest untouched
        self.assertEqual(len(result.positions["MSFT"]["lots"]), 2)  # 2 lots remain

        remaining_lots = result.positions["MSFT"]["lots"]

        # Middle lot should be reduced from 20 to 5 shares (25 - 10 from oldest = 15 from middle)
        self.assertEqual(remaining_lots[0].date, "2025-09-15")  # Middle lot
        self.assertEqual(remaining_lots[0].shares, 5)  # 20 - 15 = 5
        self.assertEqual(remaining_lots[0].price_per_share, 110.0)

        # Newest lot should be unchanged
        self.assertEqual(remaining_lots[1].date, "2025-10-01")  # Newest lot
        self.assertEqual(remaining_lots[1].shares, 15)  # Unchanged
        self.assertEqual(remaining_lots[1].price_per_share, 120.0)

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_8_sell_fifo_multiple_lots_entire_position(self, mock_get_latest_price):
        """Test 8: FIFO sell entire position across all lots"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [TradeRecommendation(action="SELL", symbol="MSFT", shares=45, agent_estimated_price=200.0, reasoning="Sell all MSFT")]

        result = self.multi_lot_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Verify cash increased by sale proceeds
        self.assertGreater(result.cash, 2000.0)

        # Verify MSFT position is completely removed
        self.assertNotIn("MSFT", result.positions)

        # Verify no positions remain
        self.assertEqual(len(result.positions), 0)

    @patch("stock_history_database.StockHistoryDatabase.get_latest_price")
    def test_9_mixed_buy_sell_operations(self, mock_get_latest_price):
        """Test 9: Mixed BUY and SELL operations in single call"""
        mock_get_latest_price.side_effect = self.mock_get_price

        trades = [
            TradeRecommendation(action="SELL", symbol="AAPL", shares=5, agent_estimated_price=150.0, reasoning="Partial AAPL sale"),
            TradeRecommendation(action="BUY", symbol="NVDA", shares=8, agent_estimated_price=100.0, reasoning="Buy NVDA"),
            TradeRecommendation(action="SELL", symbol="GOOG", shares=5, agent_estimated_price=300.0, reasoning="Sell all GOOG"),
        ]

        result = self.initial_portfolio.apply_trades(trades)

        # Verify function returns a portfolio (not None)
        self.assertIsNotNone(result)

        # Calculate expected cash: start 1000 + sell AAPL (5*150) + sell GOOG (5*510) - buy NVDA (8*100)
        # 1000 + 750 + 2550 - 800 = 3500
        expected_cash = 1000.0 + (5 * 150.0) + (5 * 510.0) - (8 * 100.0)
        self.assertEqual(result.cash, expected_cash)

        # Verify AAPL position reduced to 5 shares
        self.assertEqual(result.positions["AAPL"]["shares"], 5)
        self.assertEqual(result.positions["AAPL"]["current_price"], 150.0)

        # Verify GOOG position completely removed
        self.assertNotIn("GOOG", result.positions)

        # Verify NVDA position added
        self.assertIn("NVDA", result.positions)
        self.assertEqual(result.positions["NVDA"]["shares"], 8)
        self.assertEqual(result.positions["NVDA"]["current_price"], 100.0)
        self.assertEqual(len(result.positions["NVDA"]["lots"]), 1)

        # Verify total positions: AAPL + NVDA = 2
        self.assertEqual(len(result.positions), 2)


if __name__ == "__main__":
    # Setup logging
    config.setup_logging()

    unittest.main()
