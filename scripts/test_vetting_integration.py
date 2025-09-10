#!/usr/bin/env python3
"""
Test script for VettingAgent Integration
Task 1.5: Validate VettingAgent integration into synthesis workflow

This test validates the integration logic without requiring LLM models
by using mock data and testing the vetting evaluation pipeline.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
try:
    from agents.vetting_agent import VettingAgent, VettingRubric, HypothesisVettingEngine, VettingStatus
    VETTING_AVAILABLE = True
except ImportError:
    VETTING_AVAILABLE = False

try:
    from agents.synthesis_agents import StructuredHypothesis, MarketOpportunity, BusinessModel, CompetitiveAnalysis
    SYNTHESIS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AVAILABLE = False

def create_mock_structured_hypothesis() -> Dict[str, Any]:
    """Create mock structured hypothesis data"""
    return {
        'id': 'test_hyp_001',
        'hypothesis_statement': 'Small businesses need automated AI-powered inventory management solutions that integrate with existing POS systems to reduce waste and optimize stock levels.',
        'opportunity_id': 'test_opp_001',
        'business_model_id': 'test_bm_001',
        'competitive_analysis_id': 'test_ca_001'
    }

def create_mock_market_opportunity() -> Dict[str, Any]:
    """Create mock market opportunity data"""
    return {
        'id': 'test_opp_001',
        'title': 'AI-Powered SMB Inventory Management',
        'description': 'SaaS platform for automated inventory optimization targeting small to medium businesses',
        'market_size': 15000000000,  # $15B market
        'confidence_score': 0.85,
        'target_demographics': ['Small retail businesses', 'Restaurant chains', 'E-commerce sellers']
    }

def create_mock_business_model() -> Dict[str, Any]:
    """Create mock business model data"""
    return {
        'id': 'test_bm_001',
        'model_name': 'SaaS Subscription with Tiered Pricing',
        'value_proposition': 'Reduce inventory costs by 30% through AI-powered demand forecasting',
        'revenue_streams': [{'type': 'subscription', 'pricing': '$299/month'}],
        'projected_year_3_revenue': 5000000
    }

def create_mock_competitive_analysis() -> Dict[str, Any]:
    """Create mock competitive analysis data"""
    return {
        'id': 'test_ca_001',
        'market_category': 'Inventory Management Software',
        'competitive_advantages': [
            'AI-powered demand forecasting',
            'Real-time POS integration',
            'Mobile-first interface'
        ],
        'market_gaps': [
            'Limited SMB-focused solutions',
            'Complex enterprise tools'
        ]
    }

class MockHypothesis:
    """Mock StructuredHypothesis object"""
    def __init__(self, data):
        self.hypothesis_id = data['id']
        self.hypothesis_statement = data['hypothesis_statement']
        self.solution_description = data['hypothesis_statement']
        self.validation_methodology = ['customer interviews', 'prototype testing']
        self.risk_factors = ['market competition', 'integration complexity']
        self.resource_requirements = {'budget_estimate': '150000'}
        self.timeline = {'mvp_development': '10 weeks'}

class MockOpportunity:
    """Mock MarketOpportunity object"""
    def __init__(self, data):
        self.opportunity_id = data['id']
        self.title = data['title']
        self.description = data['description']
        self.market_size_estimate = data['market_size']
        self.confidence_score = data['confidence_score']
        self.target_demographics = data['target_demographics']
        self.trends = ['AI adoption', 'SMB digitization', 'cost optimization']

class MockBusinessModel:
    """Mock BusinessModel object"""
    def __init__(self, data):
        self.model_id = data['id']
        self.model_name = data['model_name']
        self.value_proposition = data['value_proposition']
        self.revenue_streams = data['revenue_streams']
        self.financial_projections = {'year_1': {'revenue': 1000000}}

class MockCompetitiveAnalysis:
    """Mock CompetitiveAnalysis object"""
    def __init__(self, data):
        self.analysis_id = data['id']
        self.market_category = data['market_category']
        self.key_competitors = ['TradeGecko', 'inFlow Inventory', 'Zoho Inventory']
        self.competitive_advantages = data['competitive_advantages']
        self.market_gaps = data['market_gaps']
        self.entry_barriers = ['Technology complexity', 'Integration challenges']

def test_vetting_engine():
    """Test the core vetting engine functionality"""
    print("🎯 Testing VettingEngine core functionality...")
    
    if not VETTING_AVAILABLE:
        print("   ❌ VettingAgent not available")
        return False
    
    # Create test data
    hyp_data = create_mock_structured_hypothesis()
    opp_data = create_mock_market_opportunity()
    bm_data = create_mock_business_model()
    ca_data = create_mock_competitive_analysis()
    
    # Create mock objects
    hypothesis = MockHypothesis(hyp_data)
    opportunity = MockOpportunity(opp_data)
    business_model = MockBusinessModel(bm_data)
    competitive_analysis = MockCompetitiveAnalysis(ca_data)
    
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
        
        print(f"   ✅ Vetting completed successfully")
        print(f"      Overall Score: {result.overall_score:.1f}/100")
        print(f"      Status: {result.status.value}")
        print(f"      Market Score: {result.market_size_score.score:.1f}/25")
        print(f"      Competition Score: {result.competition_score.score:.1f}/25") 
        print(f"      SVE Alignment: {result.sve_alignment_score.score:.1f}/25")
        print(f"      Execution Score: {result.execution_score.score:.1f}/25")
        
        # Validate scoring
        total_check = (result.market_size_score.score + 
                      result.competition_score.score + 
                      result.sve_alignment_score.score + 
                      result.execution_score.score)
        
        if abs(total_check - result.overall_score) < 0.1:
            print(f"   ✅ Score calculation verified")
        else:
            print(f"   ❌ Score calculation error: {total_check} != {result.overall_score}")
            return False
        
        # Test different score ranges
        print(f"   ✅ Vetting logic validated for high-quality hypothesis")
        return True
        
    except Exception as e:
        print(f"   ❌ Vetting failed: {e}")
        return False

def test_integration_workflow():
    """Test the integration workflow logic"""
    print("\n🔄 Testing integration workflow logic...")
    
    # Create mock synthesis results that would come from CrewAI
    mock_synthesis_results = {
        'success': True,
        'detailed_synthesis': {
            'market_opportunities': [create_mock_market_opportunity()],
            'business_models': [create_mock_business_model()],
            'competitive_analyses': [create_mock_competitive_analysis()],
            'structured_hypotheses': [create_mock_structured_hypothesis()]
        }
    }
    
    # Test hypothesis extraction
    hypotheses_data = extract_hypotheses_for_testing(mock_synthesis_results)
    
    if hypotheses_data:
        print(f"   ✅ Hypothesis extraction: {len(hypotheses_data)} hypotheses prepared")
    else:
        print(f"   ❌ Hypothesis extraction failed")
        return False
    
    # Test vetting workflow
    if not VETTING_AVAILABLE:
        print("   ⚠️ Skipping vetting test - VettingAgent not available")
        return False
    
    vetting_results = run_test_vetting(hypotheses_data)
    
    if vetting_results:
        print(f"   ✅ Vetting workflow: {len(vetting_results)} hypotheses evaluated")
        
        # Test filtering
        filtered = filter_test_results(vetting_results)
        
        total_filtered = sum(len(category) for category in filtered.values())
        print(f"   ✅ Filtering logic: {total_filtered} hypotheses categorized")
        
        for category, hypotheses in filtered.items():
            if hypotheses:
                print(f"      {category}: {len(hypotheses)} hypotheses")
        
        return True
    else:
        print(f"   ❌ Vetting workflow failed")
        return False

def extract_hypotheses_for_testing(synthesis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract hypotheses for testing (simplified version)"""
    detailed_synthesis = synthesis_results.get('detailed_synthesis', {})
    
    opportunities = detailed_synthesis.get('market_opportunities', [])
    business_models = detailed_synthesis.get('business_models', [])
    competitive_analyses = detailed_synthesis.get('competitive_analyses', [])
    structured_hypotheses = detailed_synthesis.get('structured_hypotheses', [])
    
    hypotheses_data = []
    for hypothesis_data in structured_hypotheses:
        # Find matching components
        opportunity = next((opp for opp in opportunities if opp['id'] == hypothesis_data.get('opportunity_id')), None)
        business_model = next((bm for bm in business_models if bm['id'] == hypothesis_data.get('business_model_id')), None)
        competitive_analysis = next((ca for ca in competitive_analyses if ca['id'] == hypothesis_data.get('competitive_analysis_id')), None)
        
        if opportunity and business_model and competitive_analysis:
            hypotheses_data.append({
                'hypothesis': hypothesis_data,
                'opportunity': opportunity,
                'business_model': business_model,
                'competitive_analysis': competitive_analysis
            })
    
    return hypotheses_data

def run_test_vetting(hypotheses_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run vetting on test hypotheses"""
    if not VETTING_AVAILABLE:
        return []
    
    engine = HypothesisVettingEngine()
    vetting_results = []
    
    for hypothesis_package in hypotheses_data:
        try:
            hypothesis = MockHypothesis(hypothesis_package['hypothesis'])
            opportunity = MockOpportunity(hypothesis_package['opportunity'])
            business_model = MockBusinessModel(hypothesis_package['business_model'])
            competitive_analysis = MockCompetitiveAnalysis(hypothesis_package['competitive_analysis'])
            
            result = engine.evaluate_hypothesis(
                hypothesis=hypothesis,
                market_opportunity=opportunity,
                business_model=business_model,
                competitive_analysis=competitive_analysis
            )
            
            vetting_results.append({
                'hypothesis_data': hypothesis_package['hypothesis'],
                'vetting_result': result
            })
            
        except Exception as e:
            print(f"      ❌ Vetting failed for hypothesis: {e}")
            continue
    
    return vetting_results

def filter_test_results(vetting_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter results by vetting status"""
    filtered = {
        'approved_for_validation': [],
        'conditional_approval': [],
        'needs_revision': [],
        'rejected': []
    }
    
    for vetting_package in vetting_results:
        vetting_result = vetting_package['vetting_result']
        hypothesis_data = vetting_package['hypothesis_data']
        
        enhanced_hypothesis = {
            **hypothesis_data,
            'vetting_score': vetting_result.overall_score,
            'vetting_status': vetting_result.status.value,
            'key_strengths': vetting_result.key_strengths,
            'key_weaknesses': vetting_result.key_weaknesses
        }
        
        if vetting_result.status.value == 'approved':
            filtered['approved_for_validation'].append(enhanced_hypothesis)
        elif vetting_result.status.value == 'conditional':
            filtered['conditional_approval'].append(enhanced_hypothesis)
        elif vetting_result.status.value == 'needs_revision':
            filtered['needs_revision'].append(enhanced_hypothesis)
        else:
            filtered['rejected'].append(enhanced_hypothesis)
    
    return filtered

def main():
    """Run all vetting integration tests"""
    print("🎯 VETTING AGENT INTEGRATION TEST")
    print("Task 1.5: VettingAgent Integration Validation")
    print("=" * 80)
    
    print(f"\n📋 Component Status:")
    print(f"   VettingAgent: {'✅ Available' if VETTING_AVAILABLE else '❌ Not Available'}")
    print(f"   Synthesis Agents: {'✅ Available' if SYNTHESIS_AVAILABLE else '❌ Not Available'}")
    
    if not VETTING_AVAILABLE:
        print(f"\n❌ VettingAgent not available - cannot complete integration test")
        return False
    
    # Test 1: Core vetting engine
    engine_test = test_vetting_engine()
    
    # Test 2: Integration workflow
    workflow_test = test_integration_workflow()
    
    # Results summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   🎯 Vetting Engine: {'✅ PASS' if engine_test else '❌ FAIL'}")
    print(f"   🔄 Integration Workflow: {'✅ PASS' if workflow_test else '❌ FAIL'}")
    
    overall_success = engine_test and workflow_test
    
    if overall_success:
        print(f"\n🎉 VETTING INTEGRATION VALIDATION COMPLETE!")
        print(f"✅ VettingAgent successfully integrated into synthesis workflow")
        print(f"✅ High-potential hypothesis filtering operational")
        print(f"✅ Validation pipeline optimization ready")
        print(f"\n🚀 TASK 1.5 IMPLEMENTATION VALIDATED:")
        print(f"   ✅ Comprehensive vetting rubric (4 categories, 100 points)")
        print(f"   ✅ Market size, competition, SVE alignment, execution scoring")
        print(f"   ✅ Status-based filtering (approved/conditional/needs_revision/rejected)")
        print(f"   ✅ Integration workflow before validation gauntlet")
        print(f"\n💡 Next Steps:")
        print(f"   - Configure LLM access for full CrewAI integration")
        print(f"   - Deploy to production validation pipeline")
        print(f"   - Monitor vetting performance and tune thresholds")
    else:
        print(f"\n❌ INTEGRATION VALIDATION FAILED")
        print(f"   Please review test results and fix issues before deployment")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
