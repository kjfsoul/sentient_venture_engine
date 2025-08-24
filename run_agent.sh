#!/bin/bash

# SVE Market Intelligence Agent Runner
# This script activates the conda environment and runs the market intelligence agent

# Set the project directory
PROJECT_DIR="/Users/kfitz/sentient_venture_engine"

# Change to project directory
cd "$PROJECT_DIR" || exit

# Activate conda environment and run the agent
source /Users/kfitz/opt/anaconda3/bin/activate sve_env

# Run the market intelligence agent
python agents/market_intel_agents.py

# Exit with the same code as the Python script
exit $?
