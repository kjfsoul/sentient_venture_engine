#!/usr/bin/env python3
"""
Test script for Enhanced Tier 1 Sentiment Analysis Agent
Tests the new functionality including Reddit API integration, web search,
LLM sentiment analysis, and secure API key management.
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test individual components to avoid import issues
try:
    from agents.validation_agents import Tier1SentimentAgent, SentimentAnalysis
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False

# Load environment variables
load_dotenv()

def test_enhanced_tier1_agent():
    """Test the enhanced Tier 1 sentiment analysis agent"""
    print("🧪 Testing Enhanced Tier 1 Sentiment Analysis Agent")
    print("=" * 60)
    
    # Test hypothesis
    test_hypothesis = {
        'hypothesis_id': 'test_hypothesis_001',
        'hypothesis_statement': 'AI-powered customer service automation platform for small businesses to reduce support costs while improving response times and customer satisfaction.',
        'market_size_estimate': '$2.5B - $5B',
        'target_audience': 'Small businesses with 10-100 employees',
        'key_value_propositions': [
            'Reduce customer support costs by 40-60%',
            'Improve response times from hours to minutes',
            '24/7 automated customer support',
            'Seamless integration with existing CRM systems'
        ]
    }
    
    # Initialize agent
    agent = Tier1SentimentAgent()
    
    # Test 1: Check API credentials
    print("\n📋 Test 1: API Credential Validation")
    credentials = agent.credentials_status
    for service, status in credentials.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {service}: {'Valid' if status else 'Invalid'}")
    
    # Test 2: Test sentiment analysis
    print("\n🎯 Test 2: Sentiment Analysis")
    try:
        sentiment_result = agent.analyze_sentiment(test_hypothesis)
        print(f"  Analysis ID: {sentiment_result.analysis_id}")
        print(f"  Overall Sentiment: {sentiment_result.overall_sentiment}")
        print(f"  Sentiment Score: {sentiment_result.sentiment_score:.3f}")
        print(f"  Market Receptivity: {sentiment_result.market_receptivity_score:.3f}")
        print(f"  Social Media Mentions: {sentiment_result.social_media_mentions}")
        print(f"  Key Positive Signals: {len(sentiment_result.key_positive_signals)}")
        print(f"  Key Negative Signals: {len(sentiment_result.key_negative_signals)}")
        
        # Print key findings
        print("\n  📊 Key Findings:")
        for i, finding in enumerate(sentiment_result.key_positive_signals[:3], 1):
            print(f"    {i}. {finding}")
        for i, finding in enumerate(sentiment_result.key_negative_signals[:2], 1):
            print(f"    ⚠️  {i}. {finding}")
            
    except Exception as e:
        print(f"  ❌ Sentiment analysis failed: {e}")
        return False
    
    # Test 3: Test storage (if Supabase is available)
    print("\n💾 Test 3: Storage Test")
    if agent.supabase:
        try:
            storage_result = agent.store_sentiment_analysis(sentiment_result)
            if storage_result:
                print("  ✅ Sentiment analysis stored successfully")
            else:
                print("  ❌ Failed to store sentiment analysis")
        except Exception as e:
            print(f"  ❌ Storage test failed: {e}")
    else:
        print("  ⚠️  Supabase not available - skipping storage test")
    
    # Test 4: Test error handling
    print("\n🛡️  Test 4: Error Handling")
    print("  Testing error tracking...")
    print(f"  Reddit API errors: {agent.error_counts['reddit_api_errors']}")
    print(f"  Web search errors: {agent.error_counts['web_search_errors']}")
    print(f"  LLM errors: {agent.error_counts['llm_errors']}")
    print(f"  Supabase errors: {agent.error_counts['supabase_errors']}")
    
    # Test 5: Test rate limiting
    print("\n⏱️  Test 5: Rate Limiting")
    print("  Rate limiter initialized:")
    print(f"  Reddit: {agent.reddit_limiter.calls_per_second} calls/sec, max {agent.reddit_limiter.max_calls}")
    print(f"  Web: {agent.web_limiter.calls_per_second} calls/sec, max {agent.web_limiter.max_calls}")
    print(f"  LLM: {agent.llm_limiter.calls_per_second} calls/sec, max {agent.llm_limiter.max_calls}")
    
    print("\n🎉 Enhanced Tier 1 Agent Test Complete!")
    return True

def test_api_key_validation():
    """Test API key validation functionality"""
    print("\n🔐 Test: API Key Validation")
    print("=" * 40)
    
    agent = Tier1SentimentAgent()
    
    # Test secure environment variable retrieval
    test_keys = [
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET', 
        'OPENROUTER_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]
    
    for key in test_keys:
        value = agent._get_secure_env_var(key)
        status = "✅ Valid" if value else "❌ Missing/Invalid"
        print(f"  {key}: {status}")
    
    # Test credential validation
    credentials = agent._validate_api_credentials()
    print(f"\n  Overall credential status: {credentials}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Enhanced Tier 1 Agent Tests")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests due to import issues")
        print("This is likely due to dependency conflicts in the environment")
        print("The enhanced agent has been implemented but cannot be tested in this environment")
        sys.exit(1)
    
    # Run tests
    test_api_key_validation()
    test_enhanced_tier1_agent()
    
    print("\n✅ All tests completed!")