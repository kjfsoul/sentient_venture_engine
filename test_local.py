#!/usr/bin/env python3
"""
Local test script for the market intelligence agent.
This runs in test mode to avoid API rate limits and costs.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_market_intelligence():
    """Test the market intelligence system in local mode."""
    print("🧪 Starting Local Test Mode...")
    
    # Set test mode environment variable
    os.environ["TEST_MODE"] = "true"
    
    try:
        # Import and run the market analysis
        from agents.market_intel_agents import run_market_analysis
        
        print("📊 Running market analysis in test mode...")
        run_market_analysis()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Sentient Venture Engine - Local Test")
    print("=" * 50)
    
    success = test_market_intelligence()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        print("💡 To run in production mode, set TEST_MODE=false in .env")
    else:
        print("❌ Tests failed!")
        print("🔧 Check the error messages above")
    print("=" * 50)
