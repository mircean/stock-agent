"""
Portfolio Database Module
Manages historical portfolio performance tracking.
"""

import logging
import os
from typing import Dict, List

import config
import psycopg2

logger = logging.getLogger(__name__)


class PortfolioDatabase:
    """Database manager for historical portfolio performance tracking"""

    def __init__(self, cfg: config.Config):
        # PostgreSQL connection parameters
        self.db_host = os.getenv("PORTFOLIO_DB_HOST", "localhost")
        self.db_user = os.getenv("PORTFOLIO_DB_USER", "Mircea")
        self.db_password = os.getenv("PORTFOLIO_DB_PASSWORD", "")  # No password needed
        self.db_name = cfg.portfolio_db_name
        self.init_database()

    def init_database(self):
        """Initialize portfolio database with required tables and stored procedures"""
        schema_path = os.path.join(os.path.dirname(__file__), "portfolio_schema.sql")
        sprocs_path = os.path.join(os.path.dirname(__file__), "portfolio_sprocs.sql")

        # First ensure database exists
        self._create_database_if_not_exists()

        # Connect to portfolio database and run scripts
        conn_params = {"host": self.db_host, "user": self.db_user, "database": self.db_name}
        if self.db_password:
            conn_params["password"] = self.db_password
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Run schema script
                with open(schema_path, "r") as f:
                    schema_sql = f.read()
                cursor.execute(schema_sql)

                # Run stored procedures script
                with open(sprocs_path, "r") as f:
                    sprocs_sql = f.read()
                cursor.execute(sprocs_sql)

                conn.commit()
                logger.info("Portfolio database initialized with tables and stored procedures")

    def _create_database_if_not_exists(self):
        """Create portfolio database if it doesn't exist"""
        try:
            # Connect to default postgres database
            conn_params = {"host": self.db_host, "user": self.db_user, "database": "postgres"}
            if self.db_password:
                conn_params["password"] = self.db_password
            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True

            with conn.cursor() as cursor:
                cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{self.db_name}'")
                if not cursor.fetchone():
                    cursor.execute(f"CREATE DATABASE {self.db_name}")
                    logger.info(f"Created PostgreSQL database: {self.db_name}")

            conn.close()
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise

    def _get_connection(self):
        """Get database connection"""
        conn_params = {"host": self.db_host, "user": self.db_user, "database": self.db_name}
        if self.db_password:
            conn_params["password"] = self.db_password
        return psycopg2.connect(**conn_params)

    @staticmethod
    def drop_database(db_name: str):
        """Drop PostgreSQL database if it exists"""
        try:
            db_host = os.getenv("PORTFOLIO_DB_HOST", "localhost")
            db_user = os.getenv("PORTFOLIO_DB_USER", "Mircea")
            db_password = os.getenv("PORTFOLIO_DB_PASSWORD", "")

            conn_params = {"host": db_host, "user": db_user, "database": "postgres"}
            if db_password:
                conn_params["password"] = db_password

            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True

            with conn.cursor() as cursor:
                # Terminate existing connections to the database
                cursor.execute(f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{db_name}'
                    AND pid <> pg_backend_pid()
                """)

                # Drop database if it exists
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                logger.info(f"Dropped PostgreSQL database: {db_name}")

            conn.close()
        except Exception as e:
            logger.warning(f"Could not drop database {db_name}: {e}")

    def save_portfolio_snapshot(self, date: str, cash: float, positions_value: float, total_value: float, position_count: int):
        """Save daily portfolio snapshot for historical tracking"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO portfolio_history
                    (date, cash, positions_value, total_value, position_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        cash = EXCLUDED.cash,
                        positions_value = EXCLUDED.positions_value,
                        total_value = EXCLUDED.total_value,
                        position_count = EXCLUDED.position_count
                    """,
                    (date, cash, positions_value, total_value, position_count),
                )

                conn.commit()
                logger.info(f"Saved portfolio snapshot for {date}: ${total_value:.2f} total value")

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio value history for the specified number of days"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT date, cash, positions_value, total_value, position_count
                    FROM portfolio_history
                    ORDER BY date DESC
                    LIMIT %s
                    """,
                    (days,),
                )

                results = cursor.fetchall()

                history = []
                for row in results:
                    history.append(
                        {
                            "date": row[0],
                            "cash": float(row[1]) if row[1] is not None else None,
                            "positions_value": float(row[2]) if row[2] is not None else None,
                            "total_value": float(row[3]) if row[3] is not None else None,
                            "position_count": row[4],
                        }
                    )

                return list(reversed(history))  # Return chronologically

    def save_nasdaq100_snapshot(self, date: str, value: float):
        """Save daily NASDAQ 100 index value for benchmarking"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO nasdaq100_history
                    (date, value)
                    VALUES (%s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        value = EXCLUDED.value
                    """,
                    (date, value),
                )

                conn.commit()
                logger.info(f"Saved NASDAQ 100 snapshot for {date}: {value:.2f}")

    def get_nasdaq100_history(self, days: int = 30) -> List[Dict]:
        """Get NASDAQ 100 index history for the specified number of days"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT date, value
                    FROM nasdaq100_history
                    ORDER BY date DESC
                    LIMIT %s
                    """,
                    (days,),
                )

                results = cursor.fetchall()

                history = []
                for row in results:
                    history.append(
                        {
                            "date": row[0],
                            "value": float(row[1]) if row[1] is not None else None,
                        }
                    )

                return list(reversed(history))  # Return chronologically


if __name__ == "__main__":
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Initialize and verify database connectivity
    portfolio_db = PortfolioDatabase()
    logger.info("✓ Portfolio database initialized")
    logger.info("")

    # Test get_portfolio_history
    logger.info("Testing get_portfolio_history()...")
    history = portfolio_db.get_portfolio_history(days=30)
    logger.info(f"✓ Retrieved {len(history)} history records")

    if history:
        logger.info("Recent history:")
        for record in history[-5:]:  # Show last 5 records
            logger.info(f"  {record['date']}: ${record['total_value']:.2f} total")
    else:
        logger.info("  No history records found")
    logger.info("")

    logger.info("All tests passed!")
