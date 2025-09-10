#!/bin/bash

# SVE Market Intelligence Agent Runner
# This script activates the conda environment and runs the unified intelligence agent

# Set the project directory
PROJECT_DIR="/Users/kfitz/sentient_venture_engine"

# Change to project directory
cd "$PROJECT_DIR" || exit

# Activate conda environment and run the agent
source /Users/kfitz/opt/anaconda3/bin/activate sve_env

# Run the unified intelligence agent
python agents/unified_intelligence_agent.py

# Exit with the same code as the Python script
exit $?
