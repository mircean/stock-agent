#!/usr/bin/env python3
"""
Unit test for email generation and sending
"""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import config
from agent import ResearchOutput, StockScore, TradeRecommendation, TradingOutput
from automation import generate_trading_email, send_email
from portfolio import Lot, Portfolio


class TestEmailGeneration(unittest.TestCase):
    """Test email generation and sending"""

    def setUp(self):
        """Set up test data"""
        # Create state dict with all required fields
        self.state = {
            "research_analysis": "Analyzed market conditions. Tech sector showing strong momentum with NVDA leading at 95.0 composite score. MSFT stable at 88.5. Market breadth is positive.",
            "memory_analysis": "Historical trends show NVDA on sustained uptrend (trend_slope: +1.2 over 10 days). MSFT showing consolidation with low volatility. No significant deterioration in holdings.",
            "trading_analysis": "This is a test trading analysis with detailed market insights and recommendations.",
            "trading_output": TradingOutput(
                summary="Test email with sample market analysis showing strong tech sector performance.",
                trade_recommendations=[
                    TradeRecommendation(action="BUY", symbol="AAPL", shares=10, agent_estimated_price=180.50, reasoning="Strong iPhone sales and services growth", confidence="HIGH"),
                    TradeRecommendation(action="SELL", symbol="GOOG", shares=5, agent_estimated_price=2850.00, reasoning="Regulatory concerns and competition in search", confidence="MEDIUM"),
                ],
                market_outlook="Bullish - Expecting continued growth in AI and cloud sectors",
                risk_assessment="Key risks: (1) Rising interest rates. (2) Regulatory scrutiny on big tech. (3) Supply chain concerns for hardware manufacturers.",
            ),
            "research_output": ResearchOutput(
                current_holdings_scores=[
                    StockScore(symbol="MSFT", composite_score=88.5, momentum_score=85.0, quality_score=92.0, technical_score=87.5, current_price=420.75),
                    StockScore(symbol="NVDA", composite_score=95.0, momentum_score=98.5, quality_score=89.5, technical_score=96.0, current_price=890.25),
                ],
                top_alternatives=[
                    StockScore(symbol="AMD", composite_score=82.0, momentum_score=85.5, quality_score=78.0, technical_score=82.5, current_price=145.30),
                    StockScore(symbol="TSLA", composite_score=79.5, momentum_score=75.0, quality_score=82.0, technical_score=82.0, current_price=385.50),
                ],
            ),
            "approval_output": None,
        }

        cfg = config.Config()
        self.portfolio = Portfolio(cfg)
        self.portfolio.cash = 2750.50
        self.portfolio.positions = {
            "MSFT": {
                "lots": [Lot(date="2025-01-01", shares=25, price_per_share=400.0)],
                "shares": 25,
                "current_price": 420.75,
                "current_value": 10518.75,
            },
            "NVDA": {
                "lots": [Lot(date="2025-01-01", shares=15, price_per_share=800.0)],
                "shares": 15,
                "current_price": 890.25,
                "current_value": 13353.75,
            },
            "GOOG": {
                "lots": [Lot(date="2025-01-01", shares=8, price_per_share=2800.0)],
                "shares": 8,
                "current_price": 2850.00,
                "current_value": 22800.00,
            },
        }
        self.portfolio.closed_lots = {}
        self.portfolio.positions_value = 10518.75 + 13353.75 + 22800.00
        self.portfolio.total_value = self.portfolio.cash + self.portfolio.positions_value

    def test_email_generation_works(self):
        """Test that email generation doesn't crash and produces output"""
        body = generate_trading_email(self.state, self.portfolio, None)

        # Basic checks
        self.assertIsInstance(body, str)
        self.assertGreater(len(body), 100)
        self.assertIn("Test email with sample market analysis showing strong tech sector performance.", body)
        self.assertIn("MSFT", body)
        self.assertIn("AAPL", body)
        self.assertIn("BUY", body)

        print("✅ Email generation test passed!")
        print(f"Generated email length: {len(body)} characters")

    def test_send_email(self):
        """Send a test trading email"""
        # Generate email
        subject = f"TEST Trading Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        body = generate_trading_email(self.state, self.portfolio, None)

        # Send email using the shared method
        send_email(subject, body)

        print("✅ Test email sent successfully!")
        print(f"Subject: {subject}")
        print(f"Body length: {len(body)} characters")


if __name__ == "__main__":
    # Setup environment and logging for test execution
    load_dotenv()
    config.setup_logging()
    unittest.main()
