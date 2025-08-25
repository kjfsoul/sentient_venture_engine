#!/usr/bin/env python3
"""
Minimal test agent for N8N troubleshooting.
Just outputs simple text with no complex processing.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def simple_test():
    """Simple test function with plain text output."""
    print("Starting simple test agent...")
    print("Agent initialized successfully")
    print("Processing test data...")
    print("Test trend: AI Growth")
    print("Test pain point: High Costs")
    print("Test completed successfully")
    print("EXECUTION_COMPLETE: Test agent finished")

if __name__ == "__main__":
    simple_test()
