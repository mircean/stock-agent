"""
Evaluation Framework

Runs the trading agent on historical dates to evaluate performance.
The idea:
    Get trading dates for the time period.
    Run multiple simulations.
    Simulate running the agent each day after closing.
    The last day the agent doesn't run, only the portfolio values are updated.

"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import agent
import chart_portfolio
import config
import memory_database
import portfolio
import portfolio_database
from stock_history_database import StockHistoryDatabase

logger = logging.getLogger(__name__)


def disable_langsmith():
    """Disable LangSmith tracing for evaluation runs"""
    os.environ["LANGSMITH_TRACING"] = "false"
    logger.info("LangSmith tracing disabled for evaluation")


def get_trading_dates(time_period: str) -> list[str]:
    """
    Get list of trading dates for the specified time period.

    Args:
        time_period: "1w", "1mo", "3mo", "6mo", "1y", or "all"

    Returns:
        List of trading dates in YYYY-MM-DD format, sorted chronologically
    """

    # Map time periods to days
    # TODO: 90 calndar days not trading days so we need datediff
    # period_days = {"1w": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "all": None}
    period_days = {"all": None}

    assert time_period in period_days, f"Invalid time_period: {time_period}. Must be one of {list(period_days.keys())}"

    # Get all available dates from stock_history.db
    with sqlite3.connect(config.STOCK_HISTORY_DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM stock_statistics_history ORDER BY date ASC")
        all_dates = [row[0] for row in cursor.fetchall()]

    assert all_dates, "No trading dates found in stock_history.db"

    # Filter by time period
    if time_period == "all":
        trading_dates = all_dates
    else:
        days = period_days[time_period]
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        trading_dates = [date for date in all_dates if date >= cutoff_date]

    logger.info(f"Found {len(trading_dates)} trading dates for period '{time_period}' (from {trading_dates[0]} to {trading_dates[-1]})")
    return trading_dates


def main():
    load_dotenv()

    # Disable LangSmith by default for evaluation
    # disable_langsmith()

    config.setup_logging()

    # Parse configuration with command line overrides
    cfg = config.parse_config()

    # Evaluation parameters
    time_period = "all"  # "1w", "1mo", "3mo", "6mo", "1y", or "all"
    num_simulations = 3  # Number of simulations to run

    # Set eval-specific paths
    cfg.stock_history_db_name = config.EVAL_STOCK_HISTORY_DB_NAME
    cfg.portfolio_file = config.EVAL_PORTFOLIO_FILE
    cfg.portfolio_db_name = config.EVAL_PORTFOLIO_DB_NAME
    cfg.memory_db_name = config.EVAL_MEMORY_DB_NAME

    trading_dates = get_trading_dates(time_period)

    logger.info("\n" + "=" * 80)
    logger.info("STARTING MULTI-SIMULATION EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Number of simulations: {num_simulations}")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Trading dates: {len(trading_dates)} days ({trading_dates[0]} to {trading_dates[-1]})")
    logger.info(f"Initial cash: ${cfg.default_cash:,.2f}")
    logger.info("=" * 80 + "\n")

    # Check if we should continue from previous results
    simulations_filename = "data/simulation_results.json"
    start_sim_id = 0

    if cfg.continue_simulation and os.path.exists(simulations_filename):
        logger.info(f"Found existing {simulations_filename}, checking for incomplete simulations...")
        with open(simulations_filename, "r") as f:
            simulations_json = json.load(f)

        # Count completed simulations
        start_sim_id = len(simulations_json["simulations"])
        assert start_sim_id <= num_simulations, "Start simulation ID must be less than number of simulations"
        logger.info(f"Resuming from simulation {start_sim_id + 1}/{num_simulations}")
    else:
        # Create new results file
        simulations_json = {
            "metadata": {
                "time_period": time_period,
                "num_simulations": num_simulations,
                "num_trading_dates": len(trading_dates),
                "start_date": trading_dates[0],
                "end_date": trading_dates[-1],
            },
            "simulations": [],
        }
        logger.info(f"Results will be saved to: {simulations_filename}")

    # Run multiple simulations
    for sim_id in range(start_sim_id, num_simulations):
        logger.info("\n" + "=" * 80)
        logger.info(f"SIMULATION {sim_id + 1}/{num_simulations}")
        logger.info("=" * 80 + "\n")

        # set random seed for this simulation
        cfg.llm_seed = sim_id

        # Clean up existing databases and portfolio
        if os.path.exists(cfg.portfolio_file):
            os.remove(cfg.portfolio_file)
        portfolio_database.PortfolioDatabase.drop_database(cfg.portfolio_db_name)

        # Copy real memory database and filter to data before first trading day
        logger.info(f"Preparing memory database: copying and filtering to before {trading_dates[0]}")
        shutil.copy(config.MEMORY_DB_NAME, cfg.memory_db_name)
        memory_db = memory_database.MemoryDatabase(cfg)
        memory_db.remove_data_after_date(trading_dates[0])
        logger.info(f"Memory database ready with historical data before {trading_dates[0]}")

        # Initialize fresh portfolio
        initial_portfolio = portfolio.Portfolio(cfg)
        initial_portfolio.prices_as_of = trading_dates[0]
        initial_portfolio.save()

        # Run agent for each trading date
        for idx, date in enumerate(trading_dates):
            logger.info(f"\n>>> Simulation {sim_id + 1}, Trading day {idx + 1}/{len(trading_dates)}: {date}")
            cfg.as_of_date = date
            # Copy and filter database for this date
            logger.info(f"Preparing database: copying and filtering to {cfg.as_of_date}")
            shutil.copy(config.STOCK_HISTORY_DB_NAME, cfg.stock_history_db_name)
            stock_db = StockHistoryDatabase(cfg)
            stock_db.remove_data_after_date(cfg.as_of_date)
            logger.info(f"Database ready with data up to {cfg.as_of_date}")

            # update the stock prices in portfolio file and database
            portfolio.Portfolio.load(cfg).update_stock_prices()

            # Run agent except the last day
            if idx < len(trading_dates) - 1:
                logger.info(f"{'>' * 70}")
                logger.info(f"Running agent for {cfg.as_of_date}")
                logger.info(f"{'>' * 70}")
                agent.main(cfg)

        # Collect full daily portfolio history from this simulation
        logger.info("Collecting daily portfolio history...")
        db = portfolio_database.PortfolioDatabase(cfg)
        portfolio_history = db.get_portfolio_history(days=len(trading_dates) + 10)  # Get all history

        # Calculate metrics
        initial_value = portfolio_history[0]["total_value"]
        final_value = portfolio_history[-1]["total_value"]
        total_return = ((final_value / initial_value) - 1) * 100

        logger.info(f"Collected {len(portfolio_history)} days of history")
        logger.info(f"Initial value: ${initial_value:,.2f}")
        logger.info(f"Final value: ${final_value:,.2f}")
        logger.info(f"Total return: {total_return:+.2f}%")

        # Convert date objects to strings for JSON serialization
        portfolio_history_json = []
        for record in portfolio_history:
            record_json = record.copy()
            record_json["date"] = str(record["date"])
            portfolio_history_json.append(record_json)

        # Add to results (use JSON format with string dates)
        simulations_json["simulations"].append(portfolio_history_json)

        # Save updated results to file
        with open(simulations_filename, "w") as f:
            json.dump(simulations_json, f, indent=2)
        logger.info(f"Saved simulation {sim_id + 1} results to {simulations_filename}")

        logger.info("\n" + "=" * 80)
        logger.info(f"SIMULATION {sim_id + 1} COMPLETED")
        logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("ALL SIMULATIONS COMPLETED")

    # Fetch NASDAQ 100 history for charting
    logger.info("Fetching NASDAQ 100 history for chart...")
    db = portfolio_database.PortfolioDatabase(cfg)
    nasdaq_history = db.get_nasdaq100_history(days=len(trading_dates) + 10)

    # Create performance chart
    logger.info("Creating performance chart...")
    chart_portfolio.create_performance_chart(simulations_json["simulations"], nasdaq_history)


if __name__ == "__main__":
    main()
