import logging
import os
from datetime import datetime, timezone

import markdown
from dotenv import load_dotenv

import agent
import config
import portfolio
import stock_history_sync
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)


def generate_trading_email(trading_analysis, start_portfolio, final_portfolio):
    """
    Generate a formatted markdown email body for trading analysis.

    Args:
        trading_analysis: TradingAnalysis object with recommendations and scores
        start_portfolio: Portfolio object
        final_portfolio: Portfolio object or None

    Returns:
        str: Formatted markdown email body
    """
    # Create formatted markdown email body using Portfolio.print() with markdown enabled
    body = start_portfolio.print("Start Portfolio", use_markdown=True)
    body += "\n\n"
    body += agent.print_analysis(trading_analysis, use_markdown=True)

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

    # Step 3: Run trading agent
    logger.info("Running trading agent...")
    trading_analysis, final_portfolio = agent.main(cfg)
    logger.info("Trading agent completed successfully")

    # Step 4: Send email
    subject = f"Daily Trading Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    body = generate_trading_email(trading_analysis, start_portfolio, final_portfolio)
    send_email(subject, body)

    logger.info("Daily trading report sent to email")

    logger.info("Daily trading automation completed")


if __name__ == "__main__":
    main()
