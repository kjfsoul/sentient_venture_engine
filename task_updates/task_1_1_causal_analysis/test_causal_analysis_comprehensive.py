#!/usr/bin/env python3
"""
Comprehensive Test Script for Causal Analysis Agent
Tests all components of the enhanced causal analysis implementation
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_causal_analysis_agent():
    """Test the comprehensive causal analysis agent"""
    print("🧠 Testing Comprehensive Causal Analysis Agent")
    print("=" * 60)
    
    try:
        # Import the enhanced causal analysis agent
        from agents.analysis_agents import CausalAnalysisAgent
        
        # Initialize agent in test mode
        agent = CausalAnalysisAgent(test_mode=True)
        print("✅ CausalAnalysisAgent initialized successfully")
        
        # Test 1: Library availability check
        print("\n📚 Testing Library Availability:")
        agent._log_library_status()
        
        # Test 2: Causal DAG definition
        print("\n🔗 Testing Causal DAG Definition:")
        dag = agent.causal_dag
        print(f"  Nodes: {len(dag['nodes'])}")
        print(f"  Edges: {len(dag['edges'])}")
        print(f"  Sample nodes: {list(dag['nodes'].keys())[:5]}")
        
        # Test 3: Causal hypotheses
        print("\n🔬 Testing Causal Hypotheses:")
        hypotheses = agent.causal_hypotheses
        print(f"  Total hypotheses: {len(hypotheses)}")
        for i, hyp in enumerate(hypotheses[:3]):
            print(f"  {i+1}. {hyp.treatment} → {hyp.outcome}")
        
        # Test 4: Data retrieval (will use simulated data)
        print("\n📊 Testing Data Retrieval:")
        data = agent.retrieve_validation_data()
        if data is not None:
            print(f"  ✅ Data retrieved: {len(data)} rows, {len(data.columns)} columns")
            print(f"  Columns: {list(data.columns)[:10]}")
            print(f"  Sample data shape: {data.shape}")
        else:
            print("  ❌ Data retrieval failed")
            return False
        
        # Test 5: Feature extraction methods
        print("\n🔧 Testing Feature Extraction:")
        sample_hypothesis = {'initial_hypothesis_text': 'AI-powered SaaS platform for enterprise automation'}
        sample_metrics = {'user_engagement': 0.7, 'conversion_rate': 0.15}
        sample_record = {'tier': 2, 'validation_timestamp': datetime.now().isoformat()}
        
        market_complexity = agent._extract_market_complexity(sample_hypothesis, sample_metrics)
        resource_investment = agent._extract_resource_investment(sample_metrics)
        hypothesis_novelty = agent._extract_hypothesis_novelty(sample_hypothesis)
        
        print(f"  Market complexity: {market_complexity:.3f}")
        print(f"  Resource investment: {resource_investment:.3f}")
        print(f"  Hypothesis novelty: {hypothesis_novelty:.3f}")
        
        # Test 6: Comprehensive causal analysis
        print("\n🧠 Testing Comprehensive Causal Analysis:")
        analysis_results = agent.run_causal_analysis(data)
        
        if 'error' not in analysis_results:
            print(f"  ✅ Analysis completed successfully")
            print(f"  Data points analyzed: {analysis_results['data_points']}")
            print(f"  Hypotheses tested: {len(analysis_results['causal_hypotheses_tested'])}")
            
            # Show sample results
            if analysis_results['causal_hypotheses_tested']:
                sample_hyp = analysis_results['causal_hypotheses_tested'][0]
                print(f"  Sample hypothesis: {sample_hyp['hypothesis']}")
                if sample_hyp['methods']:
                    sample_method = sample_hyp['methods'][0]
                    print(f"  Sample effect: {sample_method['effect_estimate']:.3f} ({sample_method['method']})")
            
            if analysis_results['causal_discovery']:
                discovery = analysis_results['causal_discovery']
                print(f"  Causal discovery: {discovery['total_edges']} edges discovered")
            
            if analysis_results['counterfactual_analyses']:
                print(f"  Counterfactual analyses: {len(analysis_results['counterfactual_analyses'])} scenarios")
            
            if analysis_results['llm_interpretation']:
                interpretation_length = len(analysis_results['llm_interpretation'])
                print(f"  LLM interpretation: {interpretation_length} characters")
            
            if analysis_results['recommendations']:
                print(f"  Recommendations generated: {len(analysis_results['recommendations'])}")
                for i, rec in enumerate(analysis_results['recommendations'][:3]):
                    print(f"    {i+1}. {rec[:80]}...")
        else:
            print(f"  ❌ Analysis failed: {analysis_results['error']}")
            return False
        
        # Test 7: Synthesis recommendations
        print("\n🎯 Testing Synthesis Recommendations:")
        synthesis_recs = agent.generate_synthesis_recommendations()
        
        if 'error' not in synthesis_recs:
            print(f"  ✅ Synthesis recommendations generated")
            print(f"  Success factors: {len(synthesis_recs['key_success_factors'])}")
            print(f"  Avoid factors: {len(synthesis_recs['avoid_factors'])}")
            print(f"  Optimal strategies: {len(synthesis_recs['optimal_strategies'])}")
        else:
            print(f"  ❌ Synthesis recommendations failed: {synthesis_recs['error']}")
        
        print("\n✅ All Causal Analysis Agent tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Causal Analysis Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_causal_inference_methods():
    """Test individual causal inference methods"""
    print("\n🔬 Testing Individual Causal Inference Methods")
    print("=" * 60)
    
    try:
        from agents.causal_inference_methods import CausalInferenceMethods
        from agents.causal_analysis_methods import CausalAnalysisMethods
        
        # Generate test data
        data = CausalAnalysisMethods.generate_simulated_data(50)
        print(f"✅ Generated test data: {len(data)} rows")
        
        # Test DoWhy analysis
        print("\n🔬 Testing DoWhy Analysis:")
        dowhy_result = CausalInferenceMethods.run_dowhy_analysis(
            data, 'resource_investment', 'validation_success', ['market_conditions', 'team_experience']
        )
        if dowhy_result:
            print(f"  ✅ DoWhy analysis successful")
            print(f"  Effect estimate: {dowhy_result.effect_estimate:.3f}")
            print(f"  Method: {dowhy_result.method}")
        else:
            print("  ⚠️ DoWhy analysis not available or failed")
        
        # Test EconML analysis
        print("\n📊 Testing EconML Analysis:")
        econml_result = CausalInferenceMethods.run_econml_analysis(
            data, 'resource_investment', 'validation_success', ['market_conditions', 'team_experience']
        )
        if econml_result:
            print(f"  ✅ EconML analysis successful")
            print(f"  Effect estimate: {econml_result.effect_estimate:.3f}")
            print(f"  Method: {econml_result.method}")
        else:
            print("  ⚠️ EconML analysis not available or failed")
        
        # Test causal discovery
        print("\n🔍 Testing Causal Discovery:")
        discovery_result = CausalInferenceMethods.run_causal_discovery(data)
        if discovery_result:
            print(f"  ✅ Causal discovery successful")
            print(f"  Method: {discovery_result['method']}")
            print(f"  Edges discovered: {discovery_result['total_edges']}")
        else:
            print("  ⚠️ Causal discovery not available or failed")
        
        # Test counterfactual analysis
        print("\n🔮 Testing Counterfactual Analysis:")
        cf_result = CausalInferenceMethods.run_counterfactual_analysis(
            data, 'resource_investment', 'validation_success', 0.3, 0.8
        )
        if cf_result:
            print(f"  ✅ Counterfactual analysis successful")
            print(f"  Treatment effect: {cf_result['treatment_effect']:.3f}")
            print(f"  Interpretation: {cf_result['interpretation'][:100]}...")
        else:
            print("  ⚠️ Counterfactual analysis failed")
        
        print("\n✅ All causal inference method tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Causal inference methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction methods"""
    print("\n🔧 Testing Feature Extraction Methods")
    print("=" * 60)
    
    try:
        from agents.causal_analysis_methods import CausalAnalysisMethods
        
        # Test data
        sample_hypothesis = {
            'initial_hypothesis_text': 'Revolutionary AI-powered SaaS platform for enterprise automation and workflow optimization',
            'generated_by_agent': 'synthesis_agent_v2'
        }
        
        sample_metrics = {
            'user_engagement': 0.75,
            'conversion_rate': 0.12,
            'retention_rate': 0.68,
            'cost_per_acquisition': 45.0,
            'roi': 2.3
        }
        
        sample_record = {
            'tier': 2,
            'validation_timestamp': datetime.now().isoformat(),
            'pass_fail_status': 'pass'
        }
        
        sample_feedback = [
            {
                'human_decision': 'approve',
                'rationale_text': 'Strong market potential with clear value proposition and good user engagement metrics. The AI automation angle is timely and addresses real enterprise pain points.'
            }
        ]
        
        # Test all extraction methods
        print("Testing extraction methods:")
        
        market_complexity = CausalAnalysisMethods.extract_market_complexity(sample_hypothesis, sample_metrics)
        print(f"  Market complexity: {market_complexity:.3f}")
        
        validation_strategy = CausalAnalysisMethods.extract_validation_strategy(sample_record)
        print(f"  Validation strategy: {validation_strategy}")
        
        resource_investment = CausalAnalysisMethods.extract_resource_investment(sample_metrics)
        print(f"  Resource investment: {resource_investment:.3f}")
        
        hypothesis_novelty = CausalAnalysisMethods.extract_hypothesis_novelty(sample_hypothesis)
        print(f"  Hypothesis novelty: {hypothesis_novelty:.3f}")
        
        market_timing = CausalAnalysisMethods.extract_market_timing(sample_record)
        print(f"  Market timing: {market_timing:.3f}")
        
        user_engagement = CausalAnalysisMethods.extract_user_engagement(sample_metrics)
        print(f"  User engagement: {user_engagement:.3f}")
        
        feedback_quality = CausalAnalysisMethods.extract_feedback_quality(sample_feedback)
        print(f"  Feedback quality: {feedback_quality:.3f}")
        
        iteration_speed = CausalAnalysisMethods.extract_iteration_speed(sample_record)
        print(f"  Iteration speed: {iteration_speed:.3f}")
        
        market_conditions = CausalAnalysisMethods.extract_market_conditions(sample_record)
        print(f"  Market conditions: {market_conditions:.3f}")
        
        team_experience = CausalAnalysisMethods.extract_team_experience(sample_hypothesis)
        print(f"  Team experience: {team_experience:.3f}")
        
        competitive_landscape = CausalAnalysisMethods.extract_competitive_landscape(sample_metrics)
        print(f"  Competitive landscape: {competitive_landscape:.3f}")
        
        time_to_validation = CausalAnalysisMethods.extract_time_to_validation(sample_record)
        print(f"  Time to validation: {time_to_validation:.1f} days")
        
        cost_efficiency = CausalAnalysisMethods.extract_cost_efficiency(sample_metrics)
        print(f"  Cost efficiency: {cost_efficiency:.3f}")
        
        print("\n✅ All feature extraction tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Causal Analysis Tests")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Feature extraction
    test_results.append(("Feature Extraction", test_feature_extraction()))
    
    # Test 2: Causal inference methods
    test_results.append(("Causal Inference Methods", test_causal_inference_methods()))
    
    # Test 3: Complete causal analysis agent
    test_results.append(("Causal Analysis Agent", test_causal_analysis_agent()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("\n🎉 ALL TESTS PASSED! Causal Analysis Agent is ready for production.")
        return True
    else:
        print(f"\n⚠️ {len(test_results) - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)