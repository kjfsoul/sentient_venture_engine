#!/usr/bin/env python3
"""
Quick test to verify the enhanced vetting agent is working
"""

import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.enhanced_vetting_agent import EnhancedVettingAgent, MarketContext
    print("✅ Enhanced Vetting Agent imported successfully")
    
    # Create mock data classes
    class MockStructuredHypothesis:
        def __init__(self):
            self.hypothesis_id = "test_hyp_001"
            self.hypothesis_statement = "AI-powered inventory management for SMBs"
            self.solution_description = "AI platform for automated inventory management"
            self.validation_methodology = ["customer interviews", "prototype testing"]
            self.risk_factors = ["market competition", "integration complexity"]
            self.resource_requirements = {"budget_estimate": "150000"}
            self.timeline = {"mvp_development": "8 weeks"}

    class MockMarketOpportunity:
        def __init__(self):
            self.market_size_estimate = "15000000000"
            self.confidence_score = 0.85
            self.target_demographics = ["Small retail businesses", "Restaurant chains"]
            self.trends = ["AI adoption", "SMB digitization"]

    class MockBusinessModel:
        def __init__(self):
            self.model_name = "SaaS Subscription with Tiered Pricing"
            self.value_proposition = "Reduce inventory costs by 30% through AI-powered demand forecasting"
            self.revenue_streams = [{"type": "subscription", "pricing": "$299/month"}]

    class MockCompetitiveAnalysis:
        def __init__(self):
            self.key_competitors = ["TradeGecko", "inFlow Inventory"]
            self.competitive_advantages = ["AI-powered forecasting", "Real-time POS integration"]
            self.market_gaps = ["SMB segment underserved", "Integration capabilities lacking"]
            self.entry_barriers = ["Technology complexity", "Integration challenges"]

    async def test_vetting():
        print("🚀 Testing Enhanced Vetting Agent...")
        
        # Create test data
        hypothesis = MockStructuredHypothesis()
        opportunity = MockMarketOpportunity()
        business_model = MockBusinessModel()
        competitive_analysis = MockCompetitiveAnalysis()
        market_context = MarketContext(
            industry="Software as a Service",
            economic_conditions="Growing",
            technology_trends=["AI/ML Adoption", "Cloud Migration"],
            regulatory_environment="Standard",
            competitive_intensity=0.6
        )
        
        # Initialize agent
        agent = EnhancedVettingAgent()
        print("✅ Enhanced Vetting Agent initialized")
        
        # Perform vetting
        result = await agent.vet_hypothesis_enhanced(
            hypothesis, opportunity, business_model, competitive_analysis, market_context
        )
        
        print(f"✅ Vetting completed successfully!")
        print(f"   Overall Score: {result.overall_score:.1f}/100")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence Level: {result.confidence_level:.2f}")
        
        return True

    if __name__ == "__main__":
        asyncio.run(test_vetting())
        print("\n🎉 Quick test completed successfully!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()