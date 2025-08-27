#!/usr/bin/env python3
"""
N8N-Compatible Multi-Modal Intelligence Agent
Clean output version for N8N workflow integration

Features:
- ANSI code cleaning for N8N compatibility
- Simplified output format
- Error handling for workflow environments
- JSON-structured results
"""

import os
import sys
import json
import re
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Import the unified agent
try:
    from agents.unified_multimodal_agent import UnifiedMultiModalAgent
except ImportError:
    print("ERROR: Failed to import unified multimodal agent")
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
            "unified_report": {
                "multi_modal_summary": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "image_content_analyzed": 5,
                    "video_content_analyzed": 3,
                    "total_content_pieces": 8,
                    "analysis_success_rate": 100.0
                },
                "unified_market_insights": {
                    "trending_products": ["smartphone", "fashion", "food"],
                    "consumer_behavior_patterns": ["browsing", "shopping", "social_sharing"],
                    "visual_trend_forecast": ["minimalist_design", "bright_colors", "user_generated_content"],
                    "market_sentiment_overview": {
                        "overall_sentiment": 0.75,
                        "sentiment_consistency": "high",
                        "sentiment_trend": "positive"
                    }
                },
                "actionable_recommendations": [
                    "Capitalize on cross-platform trends: smartphone, fashion, food - These elements are trending across both image and video content",
                    "Strong sentiment consistency across visual content types - Consider unified visual marketing strategy",
                    "Video content opportunity: Emerging video trend: unboxing (appeared in 2 videos)"
                ]
            },
            "content_summary": {
                "total_images_analyzed": 5,
                "total_videos_analyzed": 3,
                "cross_modal_trends_found": 3,
                "shared_brands_identified": 2
            },
            "stored_successfully": True,
            "execution_timestamp": datetime.now().isoformat()
        }
        
        print(format_for_n8n(sample_results))
        return
    
    try:
        # Initialize and run agent
        print("Starting Multi-Modal Intelligence Analysis")
        agent = UnifiedMultiModalAgent()
        
        # Run analysis
        results = agent.run_complete_multimodal_analysis()
        
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
            "execution_timestamp": datetime.now().isoformat(),
            "mode": "production"
        }
        print(format_for_n8n(error_output))

if __name__ == "__main__":
    main()
