import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import markdown
from dotenv import load_dotenv

import agent
import config
import portfolio
import stock_history_sync
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)


def generate_trading_email(state, start_portfolio, final_portfolio):
    """
    Generate a formatted markdown email body for trading analysis.

    Args:
        state: TradingState dict with all agent analysis
        start_portfolio: Portfolio object
        final_portfolio: Portfolio object or None

    Returns:
        str: Formatted markdown email body
    """
    # Create formatted markdown email body using Portfolio.print() with markdown enabled
    body = start_portfolio.print("Start Portfolio", use_markdown=True)
    body += "\n\n"
    body += agent.print_agent(state, use_markdown=True)

    body += "\n\n"
    if final_portfolio:
        body += final_portfolio.print("Final Portfolio", use_markdown=True)
    else:
        body += "No trades were executed."

    return body


def send_email(subject, body):
    """
    Send an email with the given subject and body.

    Args:
        subject: Email subject line
        body: Email body content (markdown format)
    """
    # Check required environment variables for email
    tenant_id = os.getenv("GRAPH_TENANT_ID")
    client_id = os.getenv("GRAPH_CLIENT_ID")
    assert tenant_id and client_id, "Missing required environment variables: GRAPH_TENANT_ID, GRAPH_CLIENT_ID"

    # Convert markdown to HTML
    html_body = markdown.markdown(body, extensions=["tables"])

    authenticator = OutlookAuthenticator(tenant_id=tenant_id, client_id=client_id, scopes=config.SCOPES)
    client = OutlookClient(authenticator)

    client.send_email(
        to="mircean@outlook.com",
        subject=subject,
        body=html_body,
        body_type="HTML",
    )


def backup_data(cfg):
    """
    Backup all data files to OneDrive.

    Args:
        cfg: Config object with backup_dir setting
    """
    assert cfg.backup_dir, "Backup directory must be configured"
    backup_dir = Path(cfg.backup_dir)
    assert backup_dir.exists(), f"Backup directory does not exist: {backup_dir}"

    # Create timestamped backup filename
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    backup_filename = f"stock-agent-backup_{timestamp}.zip"
    backup_path = backup_dir / backup_filename

    # Get data directory path
    data_dir = Path(__file__).parent / "data"

    logger.info(f"Creating backup: {backup_path}")

    # Create zip file with all data
    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                # Add file to zip with relative path
                zipf.write(file_path, arcname=file_path.name)
                logger.debug(f"  Added: {file_path.name}")

    # Get backup size
    backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
    logger.info(f"Backup completed successfully: {backup_filename} ({backup_size_mb:.2f} MB)")


def main():
    """Main automation function."""
    load_dotenv()

    # Parse configuration with command line overrides
    cfg = config.parse_config()

    config.setup_logging()

    logger.info("Starting daily trading automation")

    # Step 1: Sync data (optional)
    if cfg.skip_data_download:
        logger.info("Skipping data synchronization (--skip-data-download flag set)")
    else:
        logger.info("Running data synchronization...")
        stock_history_sync.main()
        logger.info("Data sync completed successfully")

    # Step 2: Update portfolio values with latest prices and save benchmark snapshot
    logger.info("Updating portfolio with latest stock prices...")
    start_portfolio = portfolio.Portfolio.load(cfg)
    start_portfolio.update_stock_prices()

    # Step 2.1: Check if market data is current (unless configured to run when market is closed)
    if not cfg.run_when_market_closed:
        start_portfolio = portfolio.Portfolio.load(cfg)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if start_portfolio.prices_as_of != today:
            logger.info(f"Market data is not from today (latest: {start_portfolio.prices_as_of}, today: {today}). Skipping agent run.")
            logger.info("Set --run-when-market-closed to override this behavior.")
            return

    # Step 3: Run trading agent
    logger.info("Running trading agent...")
    final_state, final_portfolio = agent.main(cfg)
    logger.info("Trading agent completed successfully")

    # Step 4: Backup data
    logger.info("Backing up data...")
    backup_data(cfg)

    # Step 5: Send email
    subject = f"Daily Trading Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    body = generate_trading_email(final_state, start_portfolio, final_portfolio)
    send_email(subject, body)

    logger.info("Daily trading report sent to email")

    logger.info("Daily trading automation completed")


if __name__ == "__main__":
    main()
