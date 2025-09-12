#!/bin/bash

# Simple wrapper script for N8N execution of unified intelligence agent
# This avoids conda alias issues by using direct Python path

cd /Users/kfitz/sentient_venture_engine || exit

# Use direct path to conda environment Python
/Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/unified_intelligence_n8n.py

echo "Unified Intelligence Agent execution completed with exit code: $?"
