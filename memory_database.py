#!/usr/bin/env python3
"""
Enhanced Memory Database for Trading Agent

Provides advanced analytics for agent's historical analysis and trading confidence.
"""

import json
import logging
import os
import re
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# Increment this when adding new migrations
CURRENT_SCHEMA_VERSION = 2


class MemoryDatabase:
    """Enhanced database manager for agent memory storage and advanced analytics"""

    def __init__(self, cfg: config.Config):
        self.db_path = cfg.memory_db_name
        self.cfg = cfg
        self.init_database()

    def init_database(self):
        """Initialize memory database with versioned migrations from SQL file"""
        schema_path = os.path.join(os.path.dirname(__file__), "memory_schema.sql")
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        # Parse SQL file into versioned sections
        sections = self._parse_schema_sections(schema_sql)

        # Get current version (need to run version 0 first to create schema_version table)
        current_version = self._get_schema_version(sections)

        # Apply each section where version > current_version
        with sqlite3.connect(self.db_path) as conn:
            for version, sql in enumerate(sections):
                if version > current_version:
                    logger.info(f"Applying memory database migration v{version}")
                    conn.executescript(sql)
                    conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (version,))
                    conn.commit()

    def _parse_schema_sections(self, schema_sql: str) -> List[str]:
        """Parse SQL file into versioned sections split by '-- === VERSION N ===' markers"""
        parts = re.split(r'--\s*===\s*VERSION\s+\d+\s*===', schema_sql)
        # First part is header comments, rest are version sections
        return [part.strip() for part in parts[1:] if part.strip()]

    def _get_schema_version(self, sections: List[str]) -> int:
        """Get current schema version, running version 0 first if needed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if schema_version table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            if not cursor.fetchone():
                # Run version 0 to create tables
                conn.executescript(sections[0])
                conn.execute("INSERT INTO schema_version (version) VALUES (0)")
                conn.commit()
                return 0

            cursor.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()[0]
            return result if result is not None else 0

    def update_memory(self, date: str, holdings_scores: List, alternatives_scores: List, prices: Dict, run_time: Optional[str] = None):
        """Save stock scores for holdings and alternatives.

        Args:
            date: Date in YYYY-MM-DD format
            holdings_scores: List of StockScore objects for current holdings
            alternatives_scores: List of StockScore objects for alternatives
            prices: Dict mapping symbol to current price
            run_time: Optional timestamp for this run (defaults to current time in HH:MM:SS format)
        """
        if run_time is None:
            run_time = datetime.now(timezone.utc).strftime("%H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Save holdings scores
            for score in holdings_scores:
                cursor.execute(
                    """
                    INSERT INTO agent_scores
                    (date, symbol, run_time, composite_score, momentum_score, quality_score,
                     technical_score, current_price, is_holding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date,
                        score.symbol,
                        run_time,
                        score.composite_score,
                        score.momentum_score,
                        score.quality_score,
                        score.technical_score,
                        prices[score.symbol],
                        True,  # is_holding
                    ),
                )

            # Save alternatives scores
            for score in alternatives_scores:
                cursor.execute(
                    """
                    INSERT INTO agent_scores
                    (date, symbol, run_time, composite_score, momentum_score, quality_score,
                     technical_score, current_price, is_holding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date,
                        score.symbol,
                        run_time,
                        score.composite_score,
                        score.momentum_score,
                        score.quality_score,
                        score.technical_score,
                        prices[score.symbol],
                        False,  # is_holding
                    ),
                )

            conn.commit()
            logger.info(f"Updated memory with {len(holdings_scores)} holdings and {len(alternatives_scores)} alternatives scores for {date} at {run_time}")

    def analyze_stock_trends(self, symbol: str, days: int = 7) -> Dict:
        """Analyze score trends and stability for a specific stock"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor.execute(
                """
                SELECT date, composite_score, momentum_score, quality_score, technical_score, current_price
                FROM agent_scores
                WHERE symbol = ? AND date >= ?
                ORDER BY date ASC
                """,
                (symbol.upper(), cutoff_date),
            )

            results = cursor.fetchall()

            if not results:
                return {"symbol": symbol, "error": "No data found", "days_requested": days}

            # Calculate trends and statistics
            dates = [row[0] for row in results]
            composite_scores = [row[1] for row in results]

            avg_composite = statistics.mean(composite_scores)
            score_volatility = statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0.0

            # Calculate trend metrics
            if len(composite_scores) >= 2:
                recent_avg = statistics.mean(composite_scores[-3:]) if len(composite_scores) >= 3 else composite_scores[-1]
                early_avg = statistics.mean(composite_scores[:3]) if len(composite_scores) >= 3 else composite_scores[0]
                trend_slope = (recent_avg - early_avg) / len(composite_scores)  # Points per day
                trend_strength = abs(recent_avg - early_avg)
            else:
                trend_slope = 0.0
                trend_strength = 0.0

            return {
                "symbol": symbol,
                "days_analyzed": days,
                "records_found": len(results),
                "average_composite_score": round(avg_composite, 2),
                "current_score": composite_scores[-1],
                "score_volatility": round(score_volatility, 2),
                "trend_slope": round(trend_slope, 2),
                "trend_strength": round(trend_strength, 2),
                "date_range": f"{dates[0]} to {dates[-1]}",
            }

    def compare_portfolio_performance(self, days: int = 7) -> Dict:
        """Compare all current portfolio positions over specified period"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor.execute(
                """
                SELECT symbol, AVG(composite_score) as avg_score, COUNT(*) as record_count,
                       AVG(momentum_score) as avg_momentum, AVG(quality_score) as avg_quality,
                       AVG(technical_score) as avg_technical
                FROM agent_scores
                WHERE is_holding = 1 AND date >= ?
                GROUP BY symbol
                ORDER BY avg_score DESC
                """,
                (cutoff_date,),
            )

            results = cursor.fetchall()

            if not results:
                return {"error": "No current holdings data found", "days_requested": days}

            portfolio_analysis = []
            scores = [row[1] for row in results]
            avg_portfolio_score = statistics.mean(scores)

            for row in results:
                symbol, avg_score, record_count, avg_momentum, avg_quality, avg_technical = row

                # Get trend for this holding
                trend_data = self.analyze_stock_trends(symbol, days)

                portfolio_analysis.append(
                    {
                        "symbol": symbol,
                        "average_composite_score": round(avg_score, 2),
                        "vs_portfolio_avg": round(avg_score - avg_portfolio_score, 2),
                        "trend_slope": trend_data.get("trend_slope", 0),
                        "score_volatility": trend_data.get("score_volatility", 0),
                        "trend_strength": trend_data.get("trend_strength", 0),
                        "record_count": record_count,
                        "breakdown": {"momentum": round(avg_momentum, 2), "quality": round(avg_quality, 2), "technical": round(avg_technical, 2)},
                    }
                )

            return {
                "days_analyzed": days,
                "portfolio_average_score": round(avg_portfolio_score, 2),
                "holdings_count": len(results),
                "holdings_performance": portfolio_analysis,
                "strongest_holding": portfolio_analysis[0]["symbol"] if portfolio_analysis else None,
            }

    def find_replacement_opportunities(self, min_gap: float = 5.0, days: int = 7) -> Dict:
        """Find holdings that have clearly better alternatives available for strategic replacement"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get average scores for holdings vs alternatives
            cursor.execute(
                """
                SELECT symbol, is_holding, AVG(composite_score) as avg_score, COUNT(*) as records
                FROM agent_scores
                WHERE date >= ?
                GROUP BY symbol, is_holding
                HAVING records >= 2
                """,
                (cutoff_date,),
            )

            results = cursor.fetchall()
            holdings = {row[0]: row[2] for row in results if row[1] == 1}  # is_holding = True
            alternatives = {row[0]: row[2] for row in results if row[1] == 0}  # is_holding = False

            if not holdings or not alternatives:
                return {"error": "Insufficient data for comparison", "days_requested": days}

            # Find best alternative score
            best_alternative_score = max(alternatives.values())

            underperformers = []
            for symbol, score in holdings.items():
                gap = best_alternative_score - score
                if gap >= min_gap:
                    # Get additional trend data
                    trend_data = self.analyze_stock_trends(symbol, days)

                    underperformers.append(
                        {
                            "symbol": symbol,
                            "average_score": round(score, 2),
                            "best_alternative_score": round(best_alternative_score, 2),
                            "performance_gap": round(gap, 2),
                            "trend_slope": trend_data.get("trend_slope", 0),
                            "score_volatility": trend_data.get("score_volatility", 0),
                            "trend_strength": trend_data.get("trend_strength", 0),
                        }
                    )

            underperformers.sort(key=lambda x: x["performance_gap"], reverse=True)

            return {
                "days_analyzed": days,
                "min_gap_threshold": min_gap,
                "holdings_analyzed": len(holdings),
                "alternatives_analyzed": len(alternatives),
                "underperformers_found": len(underperformers),
                "underperformers": underperformers,
                "best_alternative_score": round(best_alternative_score, 2),
            }

    def find_stocks_to_sell(self, days: int = 7, min_score_threshold: float = 60.0) -> Dict:
        """Find holdings that should be sold due to poor fundamental performance"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get current holdings with their performance metrics
            cursor.execute(
                """
                SELECT symbol, AVG(composite_score) as avg_score,
                       AVG(momentum_score) as avg_momentum,
                       AVG(quality_score) as avg_quality,
                       AVG(technical_score) as avg_technical,
                       COUNT(*) as records
                FROM agent_scores
                WHERE is_holding = 1 AND date >= ?
                GROUP BY symbol
                HAVING records >= 2
                """,
                (cutoff_date,),
            )

            results = cursor.fetchall()

            if not results:
                return {"error": "No current holdings data found", "days_requested": days}

            sell_candidates = []

            for row in results:
                symbol, avg_score, avg_momentum, avg_quality, avg_technical, record_count = row

                # Get trend analysis for this holding
                trend_data = self.analyze_stock_trends(symbol, days)

                # Return all holdings with raw metrics (let agent evaluate issues)
                trend_slope = trend_data.get("trend_slope", 0)
                score_volatility = trend_data.get("score_volatility", 0)
                trend_strength = trend_data.get("trend_strength", 0)

                # Include all holdings for agent evaluation (no pre-filtering)
                sell_candidates.append(
                    {
                        "symbol": symbol,
                        "average_score": round(avg_score, 2),
                        "trend_slope": round(trend_slope, 2),
                        "score_volatility": round(score_volatility, 2),
                        "trend_strength": round(trend_strength, 2),
                        "score_threshold": min_score_threshold,  # For reference only
                        "breakdown": {"momentum": round(avg_momentum, 2), "quality": round(avg_quality, 2), "technical": round(avg_technical, 2)},
                    }
                )

            # Sort by average score (lowest first for potential sell candidates)
            sell_candidates.sort(key=lambda x: x["average_score"])

            return {
                "days_analyzed": days,
                "holdings_analyzed": len(results),
                "sell_candidates_found": len(sell_candidates),
                "min_score_threshold": min_score_threshold,
                "sell_candidates": sell_candidates,
            }

    def find_stocks_to_buy(self, days: int = 7, min_score_threshold: float = 75.0, top_n: int = 10) -> Dict:
        """Find best available stocks (non-holdings) when cash is available"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get non-holdings (alternatives) with strong performance
            cursor.execute(
                """
                SELECT symbol, AVG(composite_score) as avg_score,
                       AVG(momentum_score) as avg_momentum,
                       AVG(quality_score) as avg_quality,
                       AVG(technical_score) as avg_technical,
                       COUNT(*) as records
                FROM agent_scores
                WHERE is_holding = 0 AND date >= ?
                GROUP BY symbol
                HAVING records >= 2 AND avg_score >= ?
                ORDER BY avg_score DESC
                LIMIT ?
                """,
                (cutoff_date, min_score_threshold, top_n * 2),  # Get extra for filtering
            )

            results = cursor.fetchall()

            if not results:
                return {"error": "No alternative stocks meet criteria", "days_requested": days}

            buy_candidates = []

            for row in results:
                symbol, avg_score, avg_momentum, avg_quality, avg_technical, record_count = row

                # Get trend analysis
                trend_data = self.analyze_stock_trends(symbol, days)
                trend_slope = trend_data.get("trend_slope", 0)
                score_volatility = trend_data.get("score_volatility", 0)
                trend_strength = trend_data.get("trend_strength", 0)

                # Return raw metrics for agent interpretation (no hardcoded thresholds)

                buy_candidates.append(
                    {
                        "symbol": symbol,
                        "average_score": round(avg_score, 2),
                        "trend_slope": round(trend_slope, 2),
                        "score_volatility": round(score_volatility, 2),
                        "trend_strength": round(trend_strength, 2),
                        "breakdown": {"momentum": round(avg_momentum, 2), "quality": round(avg_quality, 2), "technical": round(avg_technical, 2)},
                    }
                )

            # Sort by average score (highest first) and limit to top_n
            buy_candidates.sort(key=lambda x: x["average_score"], reverse=True)
            buy_candidates = buy_candidates[:top_n]

            return {
                "days_analyzed": days,
                "alternatives_analyzed": len(results),
                "min_score_threshold": min_score_threshold,
                "buy_candidates_found": len(buy_candidates),
                "top_opportunities": buy_candidates,
            }

    def get_confidence_metrics(self, symbol: str, days: int = 7) -> Dict:
        """Get comprehensive confidence metrics for trading decisions"""
        # Get basic trend analysis
        trend_data = self.analyze_stock_trends(symbol, days)

        if "error" in trend_data:
            return trend_data

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get target stock's data
            cursor.execute(
                """
                SELECT AVG(composite_score) as avg_score, is_holding
                FROM agent_scores
                WHERE symbol = ? AND date >= ?
                GROUP BY is_holding
                """,
                (symbol.upper(), cutoff_date),
            )

            target_results = cursor.fetchall()
            if not target_results:
                return {"symbol": symbol, "error": "Stock not found in recent analysis"}

            current_score = target_results[0][0]
            is_currently_held = target_results[0][1]

            # Get other holdings scores (exclude target stock)
            cursor.execute(
                """
                SELECT AVG(composite_score) as avg_score
                FROM agent_scores
                WHERE is_holding = 1 AND symbol != ? AND date >= ?
                GROUP BY symbol
                """,
                (symbol.upper(), cutoff_date),
            )
            holdings_scores = [row[0] for row in cursor.fetchall()]

            # Get alternatives scores (exclude target stock)
            cursor.execute(
                """
                SELECT AVG(composite_score) as avg_score
                FROM agent_scores
                WHERE is_holding = 0 AND symbol != ? AND date >= ?
                GROUP BY symbol
                """,
                (symbol.upper(), cutoff_date),
            )
            alternatives_scores = [row[0] for row in cursor.fetchall()]

            # Calculate confidence metrics
            immediate_vs_alternatives = 0
            sustained_performance = 0

            if alternatives_scores:
                best_alternative = max(alternatives_scores)
                immediate_vs_alternatives = current_score - best_alternative

            if holdings_scores:
                avg_holdings_score = statistics.mean(holdings_scores)
                sustained_performance = current_score - avg_holdings_score

            # Calculate raw confidence metrics (let agent interpret)
            trend_slope = trend_data.get("trend_slope", 0)
            score_volatility = trend_data.get("score_volatility", 0)
            trend_strength = trend_data.get("trend_strength", 0)

            # Base confidence score on performance gaps and trend metrics
            confidence_factors = {
                "immediate_performance_gap": immediate_vs_alternatives,
                "sustained_performance_gap": sustained_performance,
                "trend_slope": trend_slope,
                "trend_strength": trend_strength,
                "score_stability": max(0, 10 - score_volatility),  # Higher = more stable
                "current_score": current_score,
            }

            return {
                "symbol": symbol,
                "days_analyzed": days,
                "current_average_score": round(current_score, 2),
                "currently_held": is_currently_held,
                "immediate_vs_alternatives": round(immediate_vs_alternatives, 2),
                "sustained_performance_gap": round(sustained_performance, 2),
                "trend_slope": round(trend_slope, 2),
                "trend_strength": round(trend_strength, 2),
                "score_volatility": round(score_volatility, 2),
                "confidence_factors": confidence_factors,
                "portfolio_context": {
                    "holdings_count": len(holdings_scores),
                    "alternatives_count": len(alternatives_scores),
                    "avg_holdings_score": round(statistics.mean(holdings_scores), 2) if holdings_scores else None,
                    "best_alternative_score": round(max(alternatives_scores), 2) if alternatives_scores else None,
                },
            }

    def remove_data_after_date(self, date: str):
        """
        Remove all memory data from specified date onward (useful for backtesting setup).

        Args:
            date: Date in YYYY-MM-DD format. All data from this date onward will be deleted.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove agent scores from date onward
            cursor.execute("DELETE FROM agent_scores WHERE date >= ?", (date,))
            deleted_scores = cursor.rowcount

            conn.commit()

            logger.info(f"Removed {deleted_scores} agent score records from {date} onward")


if __name__ == "__main__":
    # Test the enhanced memory database
    cfg = config.Config()
    memory_db = MemoryDatabase(cfg)
    print("Enhanced memory database initialized")

    # Test analysis methods (if data exists)
    result = memory_db.compare_portfolio_performance()
    print(json.dumps(result, indent=2))
    result = memory_db.find_replacement_opportunities()
    print(json.dumps(result, indent=2))
    result = memory_db.find_stocks_to_sell()
    print(json.dumps(result, indent=2))
    result = memory_db.find_stocks_to_buy()
    print(json.dumps(result, indent=2))
    result = memory_db.get_confidence_metrics("NVDA")
    print(json.dumps(result, indent=2))
