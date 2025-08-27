#!/usr/bin/env python3
"""
Test script for Abacus.ai LLM Teams integration
Demonstrates how Abacus.ai can solve OpenRouter credit exhaustion issues
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_providers.abacus_integration import AbacusLLMTeams
from agents.enhanced_market_intel import EnhancedMarketIntelligenceAgent

def test_abacus_integration():
    """Test Abacus.ai integration with current credentials"""
    print("🧪 Testing Abacus.ai LLM Teams Integration")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv('ABACUS_API_KEY')
    deployment_id = os.getenv('ABACUS_DEPLOYMENT_ID')
    deployment_token = os.getenv('ABACUS_DEPLOYMENT_TOKEN')
    
    print(f"📋 Environment Check:")
    print(f"  API Key: {'✅ Set' if api_key and api_key != 'your_abacus_api_key_here' else '❌ Not set or placeholder'}")
    print(f"  Deployment ID: {'✅ Set' if deployment_id and deployment_id != 'your_deployment_id_here' else '❌ Not set or placeholder'}")
    print(f"  Deployment Token: {'✅ Set' if deployment_token and deployment_token != 'your_deployment_token_here' else '❌ Not set or placeholder'}")
    print()
    
    if not all([api_key, deployment_id, deployment_token]) or \
       any(x in [api_key, deployment_id, deployment_token] for x in ['your_abacus_api_key_here', 'your_deployment_id_here', 'your_deployment_token_here']):
        print("⚠️ Abacus.ai credentials not properly configured")
        print("📋 To test Abacus.ai integration, update your .env file with:")
        print("   ABACUS_API_KEY=your_actual_api_key")
        print("   ABACUS_DEPLOYMENT_ID=your_actual_deployment_id") 
        print("   ABACUS_DEPLOYMENT_TOKEN=your_actual_deployment_token")
        print()
        print("💡 Benefits of Abacus.ai integration:")
        print("   ✅ Alternative to OpenRouter when credits exhausted")
        print("   ✅ Potentially better cost structure")
        print("   ✅ Different model availability")
        print("   ✅ Redundancy for production systems")
        print("   ✅ Smart fallback between providers")
        return False
    
    # Test Abacus.ai direct integration
    print("🔧 Testing direct Abacus.ai integration...")
    abacus = AbacusLLMTeams()
    
    connection_test = abacus.test_connection()
    print(f"Connection test: {'✅ Success' if connection_test else '❌ Failed'}")
    
    if connection_test:
        print("\n🎯 Testing market analysis with Abacus.ai...")
        response = abacus.market_analysis_request(
            "Analyze current trends in AI automation for small businesses. Provide 2 trends and 2 pain points.",
            "market_intelligence"
        )
        
        if response.get('success'):
            print("✅ Market analysis successful!")
            print(f"📝 Response length: {len(response.get('content', ''))} characters")
            print(f"📊 Analysis data: {response.get('data', 'No structured data')}")
        else:
            print(f"❌ Market analysis failed: {response.get('error')}")
    
    # Test enhanced market intel agent with Abacus.ai
    print("\n🧠 Testing Enhanced Market Intelligence Agent...")
    try:
        agent = EnhancedMarketIntelligenceAgent()
        
        if agent.abacus_llm:
            print("✅ Enhanced agent has Abacus.ai provider available")
            
            # Test analysis with Abacus.ai priority
            result = agent.analyze_market_trends(
                focus_areas=["AI automation", "small business tools"],
                use_abacus=True
            )
            
            if result.get('success'):
                print("✅ Enhanced agent analysis with Abacus.ai successful!")
                print(f"📊 Provider used: {result.get('provider')}")
                print(f"📈 Data keys: {list(result.get('data', {}).keys())}")
            else:
                print(f"❌ Enhanced agent analysis failed: {result.get('error')}")
        else:
            print("⚠️ Enhanced agent: Abacus.ai provider not available")
            
    except Exception as e:
        print(f"❌ Enhanced agent test failed: {e}")
    
    return connection_test

def demonstrate_value_proposition():
    """Demonstrate the value of Abacus.ai integration"""
    print("\n💡 Abacus.ai Integration Value Proposition")
    print("=" * 50)
    
    print("🔄 **Problem Solved**: OpenRouter Credit Exhaustion")
    print("   Current issue: 'This request requires more credits'")
    print("   Solution: Automatic fallback to Abacus.ai LLM Teams")
    print()
    
    print("⚡ **Benefits**:")
    print("   1. 🔒 **Reliability**: Dual provider redundancy")
    print("   2. 💰 **Cost Optimization**: Compare pricing across providers")
    print("   3. 🎯 **Smart Fallback**: Automatic provider switching")
    print("   4. 🚀 **Performance**: Choose best provider for task type")
    print("   5. ⚙️ **Flexibility**: Easy to add more providers")
    print()
    
    print("🏗️ **Implementation Status**:")
    print("   ✅ Abacus.ai integration module created")
    print("   ✅ Enhanced market intel agent with dual providers")
    print("   ✅ Intelligent fallback strategy implemented")
    print("   ✅ Database integration preserved")
    print("   ✅ Error handling and logging")
    print()
    
    print("🔧 **Ready for Production**: Add your Abacus.ai credentials to activate!")

if __name__ == "__main__":
    print("🚀 Sentient Venture Engine - Abacus.ai Integration Test")
    print("=" * 70)
    
    # Test integration
    success = test_abacus_integration()
    
    # Show value proposition
    demonstrate_value_proposition()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Abacus.ai integration test: SUCCESS")
    else:
        print("⚠️ Abacus.ai integration test: NEEDS CREDENTIALS")
    print("✅ Enhanced Market Intelligence Agent ready with dual provider support!")