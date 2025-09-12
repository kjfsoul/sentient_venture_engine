#!/usr/bin/env python3
"""
Test script to verify improved vetting scoring
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
try:
    from agents.vetting_agent import VettingAgent, VettingRubric, HypothesisVettingEngine, VettingStatus
    VETTING_AVAILABLE = True
except ImportError:
    VETTING_AVAILABLE = False
    print("❌ VettingAgent not available")
    sys.exit(1)

class MockHypothesis:
    """Mock StructuredHypothesis object"""
    def __init__(self):
        self.hypothesis_id = 'test_hyp_001'
        self.hypothesis_statement = 'Develop an AI-powered inventory management platform that uses machine learning algorithms to predict demand patterns, automatically reorder stock, and optimize warehouse layouts. The SaaS solution integrates with existing POS systems and e-commerce platforms through automated APIs, providing real-time analytics and actionable insights.'
        self.solution_description = 'The platform leverages advanced data analytics to identify trends, reduce waste by 40%, and increase profit margins by 25% for small to medium businesses. Built with a scalable microservices architecture, the solution can process millions of data points daily and adapt to changing market conditions through continuous learning algorithms.'
        self.validation_methodology = ['customer interviews', 'prototype testing', 'A/B testing', 'market analysis']
        self.risk_factors = ['market competition', 'integration complexity']
        self.resource_requirements = {'budget_estimate': '200000'}
        self.timeline = {'mvp_development': '10 weeks'}

class MockOpportunity:
    """Mock MarketOpportunity object"""
    def __init__(self):
        self.opportunity_id = 'test_opp_001'
        self.title = 'AI-Powered SMB Inventory Management'
        self.description = 'SaaS platform for automated inventory optimization targeting small to medium businesses'
        self.market_size_estimate = 15000000000  # $15B market
        self.confidence_score = 0.85
        self.target_demographics = ['Small retail businesses', 'Restaurant chains', 'E-commerce sellers']
        self.trends = ['AI adoption', 'SMB digitization', 'cost optimization', 'supply chain optimization']

class MockBusinessModel:
    """Mock BusinessModel object"""
    def __init__(self):
        self.model_id = 'test_bm_001'
        self.model_name = 'SaaS Subscription with Tiered Pricing'
        self.value_proposition = 'Reduce inventory costs by 30% through AI-powered demand forecasting'
        self.revenue_streams = [{'type': 'subscription', 'pricing': '$299/month'}, {'type': 'enterprise', 'pricing': 'Custom'}]
        self.financial_projections = {'year_1': {'revenue': 1000000}}

class MockCompetitiveAnalysis:
    """Mock CompetitiveAnalysis object"""
    def __init__(self):
        self.analysis_id = 'test_ca_001'
        self.market_category = 'Inventory Management Software'
        self.key_competitors = ['TradeGecko', 'inFlow Inventory', 'Zoho Inventory']
        self.competitive_advantages = [
            'AI-powered demand forecasting',
            'Real-time POS integration', 
            'Mobile-first interface',
            'Continuous learning algorithms'
        ]
        self.market_gaps = [
            'Limited SMB-focused solutions',
            'Complex enterprise tools',
            'Poor integration capabilities'
        ]
        self.entry_barriers = ['Technology complexity', 'Integration challenges', 'Data security requirements']

def test_improved_scoring():
    """Test the improved vetting scoring"""
    print("🎯 Testing Improved Vetting Scoring")
    print("=" * 50)
    
    if not VETTING_AVAILABLE:
        print("❌ VettingAgent not available")
        return False
    
    # Create test data
    hypothesis = MockHypothesis()
    opportunity = MockOpportunity()
    business_model = MockBusinessModel()
    competitive_analysis = MockCompetitiveAnalysis()
    
    # Initialize vetting engine
    engine = HypothesisVettingEngine()
    
    # Run vetting evaluation
    try:
        result = engine.evaluate_hypothesis(
            hypothesis=hypothesis,
            market_opportunity=opportunity,
            business_model=business_model,
            competitive_analysis=competitive_analysis
        )
        
        print(f"✅ Vetting completed successfully")
        print(f"   Overall Score: {result.overall_score:.1f}/100")
        print(f"   Status: {result.status.value}")
        print(f"   Market Score: {result.market_size_score.score:.1f}/25")
        print(f"   Competition Score: {result.competition_score.score:.1f}/25") 
        print(f"   SVE Alignment: {result.sve_alignment_score.score:.1f}/25")
        print(f"   Execution Score: {result.execution_score.score:.1f}/25")
        
        # Show detailed breakdown
        print(f"\n📊 Detailed Breakdown:")
        print(f"   Market Size Details: {result.market_size_score.details}")
        print(f"   Competition Details: {result.competition_score.details}")
        print(f"   SVE Alignment Details: {result.sve_alignment_score.details}")
        print(f"   Execution Details: {result.execution_score.details}")
        
        # Check if score is above approval threshold
        if result.overall_score >= 60:
            print(f"\n🎉 SUCCESS: Hypothesis achieved approval threshold!")
            print(f"   Recommendation: Proceed to validation gauntlet")
        else:
            print(f"\n⚠️  WARNING: Hypothesis below approval threshold")
            print(f"   Recommendation: Review improvement recommendations")
            
        # Show recommendations
        if result.improvement_recommendations:
            print(f"\n💡 Improvement Recommendations:")
            for i, rec in enumerate(result.improvement_recommendations, 1):
                print(f"   {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vetting failed: {e}")
        return False

if __name__ == "__main__":
    success = test_improved_scoring()
    sys.exit(0 if success else 1)
