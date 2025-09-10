#!/usr/bin/env python3
"""
N8N-compatible version of the unified intelligence agent.
This version removes all ANSI color codes and complex formatting
to prevent N8N from encountering parsing errors.
"""

import os
import sys
import json
import re
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Import the unified agent
try:
    from agents.unified_intelligence_agent import UnifiedIntelligenceAgent
except ImportError:
    print("ERROR: Failed to import unified intelligence agent")
    sys.exit(1)

# Load environment variables
load_dotenv()

def clean_ansi_codes(text: str) -> str:
    """Remove ANSI color codes for N8N compatibility"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', str(text))

def format_for_n8n(data: any) -> str:
    """Format data for clean N8N output"""
    if isinstance(data, dict):
        # Convert to clean JSON
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    else:
        return clean_ansi_codes(str(data))

def main():
    """N8N-compatible main execution"""
    
    # Test mode check
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    disable_search = os.getenv('DISABLE_SEARCH', 'false').lower() == 'true'
    
    if test_mode or disable_search:
        # Return sample data for testing/rate-limit avoidance
        sample_results = {
            "success": True,
            "mode": "test_mode",
            "unified_insights": {
                "executive_summary": {
                    "total_text_trends": 3,
                    "total_code_trends": 2,
                    "total_visual_trends": 4,
                    "cross_domain_correlations": 2,
                    "analysis_timestamp": "2025-09-03T10:30:00Z"
                },
                "market_intelligence": {
                    "trends": [
                        {"title": "AI-Powered SaaS Analytics", "summary": "SaaS companies adopting AI for predictive customer analytics.", "url": "Test Knowledge Base"},
                        {"title": "No-Code Movement Expansion", "summary": "Growing adoption of no-code platforms for business automation.", "url": "Test Knowledge Base"},
                        {"title": "Creator Economy Tools", "summary": "New platforms emerging for creator monetization and audience management.", "url": "Test Knowledge Base"}
                    ],
                    "pain_points": [
                        {"title": "SaaS Integration Complexity", "summary": "Businesses struggle with connecting multiple SaaS tools effectively.", "url": "Test Knowledge Base"},
                        {"title": "Creator Payment Delays", "summary": "Content creators face delayed payments from platform monetization.", "url": "Test Knowledge Base"},
                        {"title": "AI Model Costs", "summary": "Small businesses find AI API costs prohibitive for regular use.", "url": "Test Knowledge Base"}
                    ]
                },
                "technology_intelligence": {
                    "code_trends": {"machine-learning": 15, "serverless": 12},
                    "framework_adoption": {"react": 45, "vue": 22},
                    "language_popularity": {"python": 68, "rust": 25}
                },
                "visual_intelligence": {
                    "trending_objects": {"smartphone": 28, "laptop": 22},
                    "emerging_visual_trends": {"minimalist-design": 18, "neumorphism": 15},
                    "brand_presence": {"Apple": 35, "Google": 28},
                    "color_palettes": {"blue": 42, "green": 38}
                },
                "cross_domain_insights": [
                    "Text trend 'ai' aligns with code trend 'machine-learning'",
                    "Visual trend 'minimalist-design' aligns with text trend 'no-code'"
                ],
                "actionable_recommendations": [
                    "High-priority opportunity: 'ai' appears in 2 intelligence domains",
                    "High-priority opportunity: 'minimalist' appears in 2 intelligence domains",
                    "Market validation recommended for top text-based trends",
                    "Technology stack evaluation based on emerging code patterns",
                    "Visual brand strategy alignment with emerging design trends"
                ]
            },
            "content_summary": {
                "text_trends_analyzed": 3,
                "code_repositories_analyzed": 15,
                "visual_content_analyzed": 8,
                "cross_domain_correlations": 2
            },
            "stored_successfully": True,
            "execution_timestamp": "2025-09-03T10:30:00Z"
        }
        
        print(format_for_n8n(sample_results))
        return
    
    try:
        # Initialize and run agent
        print("Starting Unified Intelligence Analysis")
        agent = UnifiedIntelligenceAgent()
        
        # Run analysis
        results = agent.run_unified_analysis()
        
        # Clean and format output
        clean_results = {}
        for key, value in results.items():
            clean_results[key] = json.loads(json.dumps(value, default=str))
        
        # Output clean JSON for N8N
        print(format_for_n8n(clean_results))
        
    except Exception as e:
        error_output = {
            "success": False,
            "error": clean_ansi_codes(str(e)),
            "execution_timestamp": "2025-09-03T10:30:00Z",
            "mode": "production"
        }
        print(format_for_n8n(error_output))

if __name__ == "__main__":
    main()
