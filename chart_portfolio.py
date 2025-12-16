"""
Portfolio Performance Chart

Creates a chart comparing portfolio performance vs NASDAQ 100 index.
Shows normalized percentage returns from the starting date.
"""

import logging

import config
import matplotlib.pyplot as plt
import pandas as pd
from portfolio_database import PortfolioDatabase

logger = logging.getLogger(__name__)


def create_performance_chart(all_portfolio_histories: list, nasdaq_history: list, save_path: str = "portfolio_performance.png"):
    """
    Create a chart comparing multiple portfolio simulations vs NASDAQ 100 performance.

    Args:
        all_portfolio_histories: List of portfolio history lists (one per simulation)
                                Each portfolio_history is a list of dicts with date, total_value, etc.
        nasdaq_history: List of NASDAQ 100 history dicts with date, value
        save_path: Path to save the chart image
    """
    if not all_portfolio_histories or not nasdaq_history:
        logger.error("No historical data available")
        return

    num_simulations = len(all_portfolio_histories)
    logger.info(f"Creating chart for {num_simulations} simulation(s)...")

    # Convert NASDAQ to DataFrame
    nasdaq_df = pd.DataFrame(nasdaq_history)
    nasdaq_df["date"] = pd.to_datetime(nasdaq_df["date"])
    nasdaq_df = nasdaq_df.set_index("date")

    # Normalize NASDAQ (percentage change from start)
    nasdaq_normalized = (nasdaq_df["value"] / nasdaq_df["value"].iloc[0] * 100) - 100

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    if len(all_portfolio_histories) == 1:
        title = "Portfolio Performance vs NASDAQ 100"
    else:
        title = f"Portfolio Performance: {num_simulations} Simulation(s) vs NASDAQ 100"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Process and plot each simulation
    simulation_normalized = []
    for idx, portfolio_history in enumerate(all_portfolio_histories):
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df = portfolio_df.set_index("date")

        # Calculate normalized returns (percentage change from start)
        normalized = (portfolio_df["total_value"] / portfolio_df["total_value"].iloc[0] * 100) - 100
        simulation_normalized.append(normalized)

        # Plot simulation
        alpha = 0.6 if num_simulations > 1 else 1.0
        ax.plot(normalized.index, normalized.values, label=f"Simulation {idx + 1}", linewidth=2, marker="o", alpha=alpha)

    # Plot average if multiple simulations
    if num_simulations > 1:
        # Align all simulations and calculate mean
        aligned_data = pd.concat(simulation_normalized, axis=1)
        avg_values = aligned_data.mean(axis=1)
        ax.plot(avg_values.index, avg_values.values, label="Average", linewidth=3, color="black", linestyle="--", marker="x")

    # Plot NASDAQ 100
    ax.plot(nasdaq_normalized.index, nasdaq_normalized.values, label="NASDAQ 100", linewidth=2.5, marker="s", color="#A23B72")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_title("Normalized Returns (% Change from Start)", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Add performance metrics
    final_returns = [sim.iloc[-1] for sim in simulation_normalized]
    avg_return = sum(final_returns) / len(final_returns) if final_returns else 0
    nasdaq_return = nasdaq_normalized.iloc[-1]
    outperformance = avg_return - nasdaq_return

    if num_simulations == 1:
        metrics_text = f"Portfolio: {final_returns[0]:+.2f}%  |  NASDAQ 100: {nasdaq_return:+.2f}%  |  Outperformance: {outperformance:+.2f}%"
    else:
        std_dev = pd.Series(final_returns).std()
        metrics_text = f"Avg Portfolio: {avg_return:+.2f}% (±{std_dev:.2f}%)  |  NASDAQ 100: {nasdaq_return:+.2f}%  |  Outperformance: {outperformance:+.2f}%"

    ax.text(0.5, 0.98, metrics_text, transform=ax.transAxes, ha="center", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=10)

    # Format dates on x-axis
    fig.autofmt_xdate()

    # Adjust layout
    plt.tight_layout()

    # Display chart
    plt.show()

    # Print summary statistics
    first_portfolio = all_portfolio_histories[0]
    first_date = first_portfolio[0]["date"]
    last_date = first_portfolio[-1]["date"]
    num_days = len(first_portfolio)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Number of simulations: {num_simulations}")
    print(f"Period: {first_date} to {last_date} ({num_days} days)")

    if num_simulations == 1:
        print("\nPortfolio:")
        print(f"  Starting Value: ${first_portfolio[0]['total_value']:,.2f}")
        print(f"  Final Value:    ${first_portfolio[-1]['total_value']:,.2f}")
        print(f"  Return:         {final_returns[0]:+.2f}%")
    else:
        print("\nSimulation Returns:")
        for i, ret in enumerate(final_returns):
            print(f"  Simulation {i + 1}: {ret:+.2f}%")
        print(f"\n  Average: {avg_return:+.2f}%")
        print(f"  Std Dev: {pd.Series(final_returns).std():.2f}%")
        print(f"  Min: {min(final_returns):+.2f}%")
        print(f"  Max: {max(final_returns):+.2f}%")

    print("\nNASDAQ 100:")
    print(f"  Return: {nasdaq_return:+.2f}%")
    print(f"\nOutperformance: {outperformance:+.2f}%")
    print("=" * 70)


def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    cfg = config.parse_config()
    # for evaluation, use the evaluation database
    # cfg.PORTFOLIO_DB_NAME = config.EVAL_PORTFOLIO_DB_NAME

    # Fetch data
    days = 365
    logger.info(f"Fetching {days} days of portfolio and NASDAQ 100 history...")
    db = PortfolioDatabase(cfg=cfg)
    portfolio_history = db.get_portfolio_history(days=days)
    nasdaq_history = db.get_nasdaq100_history(days=days)

    # Wrap in list for single simulation
    all_portfolio_histories = [portfolio_history]

    # Create chart
    create_performance_chart(all_portfolio_histories, nasdaq_history)


if __name__ == "__main__":
    main()
