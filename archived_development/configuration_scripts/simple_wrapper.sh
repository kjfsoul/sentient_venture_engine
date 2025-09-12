#!/bin/bash

# Ultra-simple wrapper to avoid N8N alias issues
echo "Starting SVE Agent..."

# Change to project directory
cd "/Users/kfitz/sentient_venture_engine" || exit

# Run the conservative agent directly
"/Users/kfitz/opt/anaconda3/envs/sve_env/bin/python" "agents/conservative_agent.py"

echo "SVE Agent completed with exit code: $?"
