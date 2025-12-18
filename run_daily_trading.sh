#!/bin/bash

# Daily Trading Agent Automation Script
# Wrapper that calls the Python daily_automation.py script

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/Users/Mircea/git/stock-agent"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Run the Python automation script
python automation.py