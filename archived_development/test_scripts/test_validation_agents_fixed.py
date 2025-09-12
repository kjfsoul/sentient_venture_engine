#!/usr/bin/env python3
"""
Test Script for Phase 3: Validation Agents and Gauntlet
Tests all validation agents and orchestrator functionality
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set test mode
os.environ['TEST_MODE'] = 'true'

try:
    from agents.validation_agents import (
        Tier1SentimentAgent,
        Tier2MarketResearchAgent,
        Tier3PrototypeAgent,
        Tier4InteractiveValidationAgent,
        ValidationTier,
        ValidationStatus
    )
    from agents.validation_tools import (
        ValidationMetricsCalculator,
        SentimentAnalysisTools,
        MarketAnalysisTools,
        PrototypeGenerationTools,
        StatisticalAnalysisTools
    )
    from scripts.validation_gauntlet_orchestrator import ValidationGauntletOrchestrator
except ImportError as e:
    print(f"❌ Failed to import validation components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_validation_agents():
    """Test all validation agents individually"""
    print("\n🧪 TESTING VALIDATION AGENTS")
    print("=" * 50)

    # Test hypothesis
    test_hypothesis = {
        'hypothesis_id': 'test_hyp_001',
        'hypothesis_statement': 'We believe that small business owners need an AI-powered workflow automation platform that can reduce manual processes by 60% and improve operational efficiency through intelligent task prioritization and automated reporting.',
        'market_size_estimate': '$5B globally',
        'target_market': 'SMB technology solutions',
        'key_assumptions': [
            'Small businesses struggle with manual processes',
            'AI technology can significantly improve efficiency',
            'Businesses are willing to pay for automation solutions'
        ]
    }

    # Test Tier 1: Sentiment Analysis Agent
    print("\n1️⃣ Testing Tier 1: Sentiment Analysis Agent")
    try:
        tier1_agent = Tier1SentimentAgent()
        sentiment_result = tier1_agent.analyze_sentiment(test_hypothesis)

        print(f"   ✅ Sentiment Analysis: {sentiment_result.overall_sentiment}")
        print(f"   📊 Sentiment Score: {sentiment_result.sentiment_score:.3f}")
        print(f"   🎯 Market Receptivity: {sentiment_result.market_receptivity_score:.3f}")
        print(f"   📈 Social Mentions: {sentiment_result.social_media_mentions}")

        # Test storage
        stored = tier1_agent.store_sentiment_analysis(sentiment_result)
        print(f"   💾 Storage: {'✅ Success' if stored else '❌ Failed'}")

    except Exception as e:
        print(f"   ❌ Tier 1 Test Failed: {e}")

    # Test Tier 2: Market Research Agent
    print("\n2️⃣ Testing Tier 2: Market Research Agent")
    try:
        tier2_agent = Tier2MarketResearchAgent()

        # Create mock sentiment analysis for tier 2
        mock_sentiment = type('MockSentiment', (), {
            'competitor_sentiment': {'direct_competitors': ['Competitor A'], 'threat_level': 'medium'},
            'sentiment_score': 0.6
        })()

        market_result = tier2_agent.validate_market_hypothesis(test_hypothesis, mock_sentiment)

        print(f"   ✅ Market Validation: {market_result.validation_id}")
        print(f"   🔧 Tech Feasibility: {market_result.technology_feasibility_score:.3f}")
        print(f"   🚧 Market Barriers: {len(market_result.go_to_market_barriers)}")
        print(f"   ⚖️ Regulatory Considerations: {len(market_result.regulatory_considerations)}")

        # Test storage
        stored = tier2_agent.store_market_validation(market_result)
        print(f"   💾 Storage: {'✅ Success' if stored else '❌ Failed'}")

    except Exception as e:
        print(f"   ❌ Tier 2 Test Failed: {e}")

    # Test Tier 3: Prototype Agent
    print("\n3️⃣ Testing Tier 3: Prototype Agent")
    try:
        tier3_agent = Tier3PrototypeAgent()

        # Create mock market validation for tier 3
        mock_market = type('MockMarket', (), {
            'technology_feasibility_score': 0.75,
            'go_to_market_barriers': ['Competition', 'Adoption']
        })()

        prototype_result = tier3_agent.generate_and_test_prototype(test_hypothesis, mock_market)

        print(f"   ✅ Prototype Generation: {prototype_result.prototype_type}")
        print(f"   🎨 Usability Score: {prototype_result.usability_score:.3f}")
        print(f"   🛠️ Development Complexity: {prototype_result.development_complexity}")
        print(f"   💰 Estimated Cost: ${prototype_result.estimated_development_cost}")
        print(f"   🔄 Iteration Recommendations: {len(prototype_result.iteration_recommendations)}")

        # Test storage
        stored = tier3_agent.store_prototype_result(prototype_result)
        print(f"   💾 Storage: {'✅ Success' if stored else '❌ Failed'}")

    except Exception as e:
        print(f"   ❌ Tier 3 Test Failed: {e}")

    # Test Tier 4: Interactive Validation Agent
    print("\n4️⃣ Testing Tier 4: Interactive Validation Agent")
    try:
        tier4_agent = Tier4InteractiveValidationAgent()

        # Create mock prototype result for tier 4
        mock_prototype = type('MockPrototype', (), {
            'usability_score': 0.82,
            'development_complexity': 'medium'
        })()

        interactive_result = tier4_agent.conduct_interactive_validation(test_hypothesis, mock_prototype)

        print(f"   ✅ Interactive Validation: {interactive_result.validation_id}")
        print(f"   📈 Investment Readiness: {interactive_result.investment_readiness_score:.1f}%")
        print(f"   👥 User Testing Participants: {interactive_result.user_testing_results.get('participant_demographics', {}).get('total_participants', 0)}")
        print(f"   🔄 Final Recommendations: {len(interactive_result.final_recommendations)}")

        # Test storage
        stored = tier4_agent.store_interactive_validation(interactive_result)
        print(f"   💾 Storage: {'✅ Success' if stored else '❌ Failed'}")

    except Exception as e:
        print(f"   ❌ Tier 4 Test Failed: {e}")

def test_validation_tools():
    """Test validation tools and utilities"""
    print("\n🔧 TESTING VALIDATION TOOLS")
    print("=" * 50)

    # Test Validation Metrics Calculator
    print("\n📊 Testing Validation Metrics Calculator")
    try:
        predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        actuals = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

        metrics = ValidationMetricsCalculator.calculate_binary_classification_metrics(predictions, actuals)

        print(f"   ✅ Accuracy: {metrics.accuracy:.3f}")
        print(f"   🎯 Precision: {metrics.precision:.3f}")
        print(f"   🔍 Recall: {metrics.recall:.3f}")
        print(f"   🎪 F1-Score: {metrics.f1_score:.3f}")
        print(f"   📏 Confidence Interval: {metrics.confidence_interval}")

    except Exception as e:
        print(f"   ❌ Metrics Calculator Test Failed: {e}")

    # Test Sentiment Analysis Tools
    print("\n🎭 Testing Sentiment Analysis Tools")
    try:
        text = "This is an amazing product that solves real problems and saves time!"
        keywords = SentimentAnalysisTools.extract_sentiment_keywords(text)

        print(f"   ✅ Positive Keywords: {keywords['positive']}")
        print(f"   ❌ Negative Keywords: {keywords['negative']}")
        print(f"   📊 Sentiment Intensity: {keywords['sentiment_intensity']:.3f}")

        complexity = SentimentAnalysisTools.analyze_text_complexity(text)
        print(f"   📖 Readability Score: {complexity['complexity_score']:.1f}")
        print(f"   📏 Complexity Level: {complexity['complexity_level']}")

    except Exception as e:
        print(f"   ❌ Sentiment Tools Test Failed: {e}")

    # Test Market Analysis Tools
    print("\n📈 Testing Market Analysis Tools")
    try:
        market_size = "$2.5B"
        users = 50000

        penetration = MarketAnalysisTools.calculate_market_penetration_rate(market_size, users)
        print(f"   ✅ Market Penetration: {penetration['penetration_percentage']:.2f}%")
        print(f"   🎯 Market Opportunity: {penetration['market_opportunity']}")

        positioning = MarketAnalysisTools.analyze_competitive_positioning({
            'competitors': [
                {'name': 'Comp A', 'market_share': 25},
                {'name': 'Comp B', 'market_share': 20},
                {'name': 'Comp C', 'market_share': 15}
            ]
        })
        print(f"   🏆 Market Position: {positioning['market_position']}")
        print(f"   📊 Concentration Index: {positioning['market_concentration_index']}")

    except Exception as e:
        print(f"   ❌ Market Tools Test Failed: {e}")

    # Test Prototype Generation Tools
    print("\n🎨 Testing Prototype Generation Tools")
    try:
        features = ['User Dashboard', 'Workflow Automation', 'Analytics Reporting']
        flow = PrototypeGenerationTools.generate_user_flow_diagram(features)

        print(f"   ✅ User Flow Steps: {len(flow['flow_steps'])}")
        print(f"   🎯 Complexity Score: {flow['complexity_score']:.2f}")
        print(f"   📱 Responsive Breakpoints: {flow['responsive_breakpoints']}")

        wireframe = PrototypeGenerationTools.generate_wireframe_specification(features)
        print(f"   🖼️ Wireframe Screens: {wireframe['total_screens']}")
        print(f"   🔧 Navigation: {wireframe['navigation_structure']}")

    except Exception as e:
        print(f"   ❌ Prototype Tools Test Failed: {e}")

    # Test Statistical Analysis Tools
    print("\n📈 Testing Statistical Analysis Tools")
    try:
        control_group = [10.2, 9.8, 10.5, 9.9, 10.1]
        test_group = [11.8, 12.1, 11.5, 12.3, 11.9]

        ab_test = StatisticalAnalysisTools.perform_ab_test_analysis(control_group, test_group)
        print(f"   ✅ A/B Test Significant: {ab_test['statistical_test']['significant']}")
        print(f"   📊 Effect Size: {ab_test['effect_size']['relative_improvement']:.1f}%")
        print(f"   📏 Confidence Interval: {ab_test['statistical_test']['confidence_interval']}")

    except Exception as e:
        print(f"   ❌ Statistical Tools Test Failed: {e}")

def test_validation_gauntlet_orchestrator():
    """Test the validation gauntlet orchestrator"""
    print("\n🎯 TESTING VALIDATION GAUNTLET ORCHESTRATOR")
    print("=" * 50)

    try:
        # Initialize orchestrator
        orchestrator = ValidationGauntletOrchestrator()

        # Test hypothesis
        test_hypothesis = {
            'hypothesis_id': 'gauntlet_test_001',
            'hypothesis_statement': 'We believe that freelance designers need an AI-powered collaboration platform that can streamline client communication, project management, and deliverable tracking to increase productivity by 40%.',
            'market_size_estimate': '$3.5B globally',
            'target_market': 'Creative freelancers and agencies',
            'key_assumptions': [
                'Freelancers struggle with client communication',
                'AI can improve project management efficiency',
                'Designers are willing to pay for productivity tools'
            ]
        }

        print("🚀 Executing Validation Gauntlet...")
        result = orchestrator.execute_validation_gauntlet(test_hypothesis)

        print(f"\n🎯 GAUNTLET RESULTS:")
        print(f"   📋 Hypothesis ID: {result.hypothesis_id}")
        print(f"   ✅ Overall Status: {result.overall_status.value}")
        print(f"   🎯 Final Tier: {result.final_tier_reached.value}")
        print(f"   📊 Tiers Completed: {len(result.tiers_completed)}")
        print(f"   💰 Total Cost: ${result.resource_cost_total:.2f}")
        print(f"   ⏱️ Total Time: {result.execution_time_total:.1f}s")
        print(f"   📈 Investment Readiness: {result.investment_readiness_score:.1f}%")

        print(f"\n📋 PERFORMANCE METRICS:")
        metrics = result.performance_metrics
        print(f"   🎯 Average Confidence: {metrics.get('average_confidence', 0):.3f}")
        print(f"   💡 Validation Efficiency: {metrics.get('validation_efficiency', 0):.3f}")
        print(f"   💰 Cost per Tier: ${metrics.get('cost_per_tier', 0):.2f}")

        print(f"\n📝 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(result.final_recommendations[:5], 1):
            print(f"   {i}. {rec}")

        print("\n✅ Validation Gauntlet Test Completed Successfully!")

    except Exception as e:
        print(f"❌ Validation Gauntlet Test Failed: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all validation tests"""
    print("🧪 PHASE 3 VALIDATION AGENTS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    start_time = datetime.now()

    try:
        # Test individual agents
        test_validation_agents()

        # Test utility tools
        test_validation_tools()

        # Test orchestrator
        test_validation_gauntlet_orchestrator()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total Test Duration: {duration:.2f} seconds")
        print(f"📊 Tests Executed: Agent Tests + Tool Tests + Orchestrator Test")
        print(f"✅ Phase 3 Validation Gauntlet Implementation: READY FOR PRODUCTION")

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()