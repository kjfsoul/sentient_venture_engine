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
    
    # Create mock data classes with all required attributes
    class MockStructuredHypothesis:
        def __init__(self):
            self.hypothesis_id = "test_hyp_001"
            self.hypothesis_statement = "AI-powered inventory management for SMBs"
            self.problem_statement = "Small businesses struggle with inventory management, leading to overstocking or stockouts"
            self.solution_description = "AI platform for automated inventory management with predictive analytics"
            self.target_customer = "Small retail businesses and restaurant chains"
            self.value_proposition = "Reduce inventory costs by 30% through AI-powered demand forecasting"
            self.key_assumptions = [
                {"assumption": "Businesses want to reduce inventory costs", "test_method": "Customer survey"},
                {"assumption": "AI can accurately predict demand", "test_method": "Prototype testing"}
            ]
            self.success_criteria = [
                {"metric": "Inventory cost reduction", "target": "30% reduction"},
                {"metric": "Stockout frequency", "target": "50% reduction"}
            ]
            self.validation_methodology = ["customer interviews", "prototype testing"]
            self.risk_factors = ["market competition", "integration complexity", "adoption resistance"]
            self.resource_requirements = {"budget_estimate": "150000"}
            self.timeline = {"mvp_development": "8 weeks"}
            self.pivot_triggers = ["No market demand", "Technical infeasibility"]
            self.test_design = {"prototype": "MVP with core features"}
            self.metrics_framework = [{"metric": "accuracy", "target": "90%"}]
            self.formulation_timestamp = "2025-09-10"
            self.validation_status = "draft"

    class MockMarketOpportunity:
        def __init__(self):
            self.opportunity_id = "opp_001"
            self.market_size_estimate = "15000000000"
            self.confidence_score = 0.85
            self.target_demographics = ["Small retail businesses", "Restaurant chains"]
            self.technology_trends = ["AI adoption", "SMB digitization"]
            self.evidence_sources = ["Market research report", "Industry analysis"]
            self.competitive_landscape = "Moderately competitive"
            self.implementation_complexity = "Medium"
            self.time_to_market = "6 months"
            self.revenue_potential = "High"
            self.risk_factors = ["Market competition", "Adoption rate"]
            self.success_metrics = ["Market share", "Revenue growth"]
            self.hypothesis_timestamp = "2025-09-10"

    class MockBusinessModel:
        def __init__(self):
            self.model_id = "bm_001"
            self.model_name = "SaaS Subscription with Tiered Pricing"
            self.value_proposition = "Reduce inventory costs by 30% through AI-powered demand forecasting"
            self.revenue_streams = [
                {"type": "subscription", "pricing": "$299/month per location"},
                {"type": "implementation", "pricing": "$5000 per business"}
            ]
            self.key_resources = ["AI algorithms", "Cloud infrastructure", "Customer support team"]
            self.key_partnerships = ["Cloud provider", "Payment processor"]
            self.implementation_roadmap = [
                {"phase": "MVP development", "timeline": "3 months"},
                {"phase": "Beta testing", "timeline": "1 month"}
            ]
            self.scalability_factors = ["Cloud-based architecture", "Automated onboarding"]
            self.risk_mitigation = ["Phased rollout", "Customer feedback integration"]
            self.channels = ["Online platform", "Direct sales"]
            self.customer_relationships = "Self-service with email support"
            self.cost_structure = {"infrastructure": "30%", "development": "40%", "marketing": "20%", "support": "10%"}
            self.financial_projections = {"year1": "$500K", "year2": "$1.2M", "year3": "$2.5M"}
            self.target_customer_segments = ["SMBs", "Retailers"]
            self.competitive_advantages = ["AI forecasting", "Ease of use"]
            self.pivot_scenarios = ["Expand to enterprise", "Add analytics"]

    class MockCompetitiveAnalysis:
        def __init__(self):
            self.analysis_id = "ca_001"
            self.competitive_advantages = ["AI-powered forecasting", "Real-time POS integration", "SMB focus"]
            self.market_gaps = ["SMB segment underserved", "Integration capabilities lacking"]
            self.differentiation_opportunities = ["Superior UX", "Better analytics"]
            self.direct_competitors = ["TradeGecko", "inFlow Inventory"]
            self.indirect_competitors = ["Manual spreadsheets", "ERP systems"]
            self.competitive_landscape = "Fragmented market with opportunities"
            self.market_positioning_map = {"x": "cost", "y": "differentiation"}
            self.competitive_disadvantages = ["Brand recognition", "Market presence"]
            self.threat_assessment = {"new_entrants": "Medium", "substitutes": "Low"}
            self.barrier_to_entry = {"capital": "Low", "regulatory": "None"}
            self.competitive_response_scenarios = ["Price war", "Feature competition"]
            self.pricing_analysis = {"premium": "20%", "competitive": "Market rate"}
            self.go_to_market_comparison = {"speed": "Faster", "coverage": "Niche"}
            self.opportunity_id = "opp_001"
            self.analysis_timestamp = "2025-09-10"

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
        print(f"   Processing Time: {result.processing_time:.2f}s")
        
        # Show some sub-factor details
        if result.sve_alignment_score.sub_factors:
            print(f"   SVE Alignment Sub-Factors: {len(result.sve_alignment_score.sub_factors)}")
            for i, factor in enumerate(result.sve_alignment_score.sub_factors[:3]):
                print(f"     {i+1}. {factor.name}: {factor.score:.1f}/10.0")
        
        return True

    if __name__ == "__main__":
        asyncio.run(test_vetting())
        print("\n🎉 Quick test completed successfully!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
