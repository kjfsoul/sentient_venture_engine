#!/usr/bin/env python3
"""
N8N-Compatible Phase 2 Synthesis Agent
Provides clean JSON output for N8N workflow integration

Orchestrates all Phase 2 agents:
- Market Opportunity Identification
- Business Model Design  
- Competitive Analysis
- Hypothesis Formulation
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the main synthesis agent
try:
    from agents.synthesis_agents import MarketOpportunityAgent
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}", "success": False}))
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise for N8N

def run_phase2_synthesis():
    """Run complete Phase 2 synthesis and return clean JSON output"""
    try:
        # Initialize the market opportunity agent
        agent = MarketOpportunityAgent()
        
        # Run the complete analysis
        results = agent.run_market_opportunity_identification()
        
        # Clean output for N8N consumption
        if results.get('success'):
            clean_output = {
                "success": True,
                "mode": "test" if os.getenv('TEST_MODE', 'false').lower() == 'true' else "production",
                "market_opportunities": results.get('market_opportunities', []),
                "business_hypotheses": results.get('business_hypotheses', []),
                "business_models": results.get('business_models', []),
                "competitive_analyses": results.get('competitive_analyses', []),
                "structured_hypotheses": results.get('structured_hypotheses', []),
                "analysis_summary": results.get('analysis_summary', {}),
                "synthesis_insights": results.get('synthesis_insights', {}),
                "timestamp": results.get('execution_timestamp')
            }
        else:
            clean_output = {
                "success": False,
                "error": results.get('error', 'Unknown error occurred'),
                "timestamp": datetime.now().isoformat()
            }
        
        return clean_output
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main execution for N8N"""
    results = run_phase2_synthesis()
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
