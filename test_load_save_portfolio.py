"""
Simple test for Portfolio class
"""

import json
import os

import config
import portfolio

# Create config
cfg = config.Config()

# Load portfolio
print("Loading portfolio...")
my_portfolio = portfolio.Portfolio.load(cfg)

print(f"Cash: ${my_portfolio.cash:.2f}")
print(f"Positions: {len(my_portfolio.positions)}")
print(f"Closed lots: {len(my_portfolio.closed_lots)}")
print(f"Total value: ${my_portfolio.total_value:.2f}")
print(f"Prices as of: {my_portfolio.prices_as_of}")

# Save portfolio to test file
print("\nSaving portfolio to test file...")
my_portfolio.cfg.portfolio_file = "portfolio_test.json"
my_portfolio.save()

# Compare both JSON files
print("\nComparing portfolio.json with portfolio_test.json...")
with open("portfolio.json", "r") as f:
    original_data = json.load(f)

with open("portfolio_test.json", "r") as f:
    test_data = json.load(f)

assert original_data == test_data, "Portfolio files do not match!"
print("✅ Files are identical")

# Clean up test file
os.remove("portfolio_test.json")
print("Cleaned up portfolio_test.json")

print("\n✅ Test passed!")
