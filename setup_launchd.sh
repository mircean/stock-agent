#!/bin/bash

# Setup script for launchd jobs
# Adds daily automation using macOS Launch Agent

PROJECT_DIR="/Users/Mircea/git/stock-agent"
PLIST_NAME="com.stockagent.dailytrading.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "Setting up launchd job for stock trading automation..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Remove existing launch agent if it exists
if launchctl list | grep -q "com.stockagent.dailytrading"; then
    echo "Unloading existing launch agent..."
    launchctl unload "$LAUNCH_AGENTS_DIR/$PLIST_NAME" 2>/dev/null || true
fi

# Copy plist to LaunchAgents directory
cp "$PROJECT_DIR/$PLIST_NAME" "$LAUNCH_AGENTS_DIR/"

# Load the launch agent
echo "Loading launch agent..."
launchctl load "$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "Launch agent installed successfully!"
echo ""
echo "Daily trading: Monday-Friday at 12:00 PM (local time)"
echo "Will run when resuming from sleep if missed during sleep"
echo ""
echo "To view status: launchctl list | grep stockagent"
echo "To unload: launchctl unload $LAUNCH_AGENTS_DIR/$PLIST_NAME"
echo "To reload: launchctl unload $LAUNCH_AGENTS_DIR/$PLIST_NAME && launchctl load $LAUNCH_AGENTS_DIR/$PLIST_NAME"
echo ""
echo "Logs will be written to:"
echo "  $PROJECT_DIR/logs/launchd.out"
echo "  $PROJECT_DIR/logs/launchd.err"