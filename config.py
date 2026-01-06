"""
Configuration for the Stock Trading Agent
"""

import argparse
import logging
from dataclasses import dataclass, fields
from typing import Final

PORTFOLIO_FILE: Final[str] = "data/portfolio.json"
PORTFOLIO_DB_NAME: Final[str] = "portfolio"
STOCK_HISTORY_DB_NAME: Final[str] = "data/stock_history.db"
MEMORY_DB_NAME: Final[str] = "data/memory.db"

# Evaluation/backtesting database and file constants
EVAL_PORTFOLIO_FILE: Final[str] = "data/portfolio_eval.json"
EVAL_PORTFOLIO_DB_NAME: Final[str] = "portfolio_eval"
EVAL_STOCK_HISTORY_DB_NAME: Final[str] = "data/stock_history_eval.db"
EVAL_MEMORY_DB_NAME: Final[str] = "data/memory_eval.db"


@dataclass
class Config:
    """Runtime configuration that can be overridden via command line."""

    # Trading limits
    max_tool_calls: int = 40
    max_positions: int = 10
    default_cash: float = 10000.0
    top_alternatives_count: int = 3  # Number of top alternative stocks to track

    # Agent behavior
    execute_trades: bool = True  # Whether to actually execute trades and update portfolio file
    skip_data_download: bool = False  # Skip data synchronization step
    run_when_market_closed: bool = False  # Run agent even if market data is not from today

    # Backup settings
    backup_dir: str = "/Users/Mircea/OneDrive/Stock Agent"

    # Evaluation parameters
    as_of_date: str = None  # Date to run agent for (YYYY-MM-DD), used for memory tracking
    continue_simulation: bool = False  # Continue from previous simulation results if available

    # Model settings
    # llm_model: str = "gpt-5.2"
    # llm_model: str = "gpt-5.2-pro"
    llm_model: str = "gemini-3-pro-preview"
    llm_temperature: float = 0.0  # Maximum determinism
    llm_seed: int = 12345  # Fixed seed for reproducible results

    portfolio_file: str = PORTFOLIO_FILE
    portfolio_db_name: str = PORTFOLIO_DB_NAME
    stock_history_db_name: str = STOCK_HISTORY_DB_NAME
    memory_db_name: str = MEMORY_DB_NAME


# Logging
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: Final[str] = "log.txt"

# Email scopes
SCOPES = ["Mail.Read", "Mail.Send"]


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser with options automatically generated from Config class."""
    parser = argparse.ArgumentParser(description="Stock Trading Agent")

    # Create default config to get default values
    default_config = Config()

    # Automatically add arguments for each config field
    for field in fields(Config):
        field_name = field.name
        field_type = field.type
        default_value = getattr(default_config, field_name)

        # Convert field_name to CLI argument (snake_case to kebab-case)
        cli_arg = "--" + field_name.replace("_", "-")

        # Set up argument based on type
        if field_type is bool:
            # For booleans, create both --flag and --no-flag options
            if default_value:
                parser.add_argument(
                    f"--no-{field_name.replace('_', '-')}", dest=field_name, action="store_false", help=f"Disable {field_name} (default: {default_value})"
                )
            else:
                parser.add_argument(cli_arg, dest=field_name, action="store_true", help=f"Enable {field_name} (default: {default_value})")
        else:
            parser.add_argument(cli_arg, dest=field_name, type=field_type, help=f"{field_name} (default: {default_value})")

    return parser


def parse_config() -> Config:
    """Parse command line arguments and return Config with overrides applied."""
    parser = create_config_parser()
    args = parser.parse_args()

    # Start with default config
    cfg = Config()

    # Override with any provided command line arguments
    for field in fields(Config):
        field_name = field.name
        if hasattr(args, field_name):
            arg_value = getattr(args, field_name)
            if arg_value is not None:
                setattr(cfg, field_name, arg_value)

    return cfg


def setup_logging():
    """Configure logging to both console and file."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger
