#!/usr/bin/env python3
"""
Phase 3: Validation Gauntlet Orchestrator
Multi-stage validation process coordinator

Manages the complete validation pipeline:
- Tier 1: Low-cost sentiment analysis
- Tier 2: Market research validation
- Tier 3: Prototype generation and testing
- Tier 4: Interactive prototype validation

Adapts validation flow based on performance at each tier and optimizes resource allocation.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Import validation agents and tools
try:
    from agents.validation_agents import (
        Tier1SentimentAgent,
        Tier2MarketResearchAgent,
        Tier3PrototypeAgent,
        Tier4InteractiveValidationAgent,
        ValidationTier,
        ValidationStatus,
        ValidationResult,
        SentimentAnalysis,
        MarketValidation,
        PrototypeResult,
        InteractiveValidation
    )
    from agents.validation_tools import (
        ValidationMetricsCalculator,
        SentimentAnalysisTools,
        MarketAnalysisTools,
        PrototypeGenerationTools,
        StatisticalAnalysisTools
    )
except ImportError as e:
    print(f"❌ Failed to import validation components: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation_gauntlet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationGauntletResult:
    """Complete result from the validation gauntlet process"""
    hypothesis_id: str
    overall_status: ValidationStatus
    tiers_completed: List[ValidationTier]
    final_tier_reached: ValidationTier
    validation_results: Dict[ValidationTier, ValidationResult]
    resource_cost_total: float
    execution_time_total: float
    investment_readiness_score: float
    final_recommendations: List[str]
    execution_timestamp: datetime
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ValidationGauntletOrchestrator:
    """Main orchestrator for the tiered validation gauntlet"""

    def __init__(self):
        # Environment configuration
        self.test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
        self.max_tier_budget = float(os.getenv('MAX_TIER_BUDGET', '1000'))
        self.stop_on_failure = os.getenv('STOP_ON_FAILURE', 'false').lower() == 'true'

        # Initialize Supabase
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = None

        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Supabase client initialized for validation gauntlet")
            except Exception as e:
                logger.error(f"❌ Supabase initialization failed: {e}")

        # Initialize validation agents
        self.tier1_agent = Tier1SentimentAgent(self.supabase)
        self.tier2_agent = Tier2MarketResearchAgent(self.supabase)
        self.tier3_agent = Tier3PrototypeAgent(self.supabase)
        self.tier4_agent = Tier4InteractiveValidationAgent(self.supabase)

        # Validation results storage
        self.validation_history = {}
        self.resource_tracking = {}

        logger.info("🎯 Validation Gauntlet Orchestrator initialized")

    def execute_validation_gauntlet(self, hypothesis: Dict[str, Any]) -> ValidationGauntletResult:
        """Execute the complete validation gauntlet for a business hypothesis"""
        logger.info(f"🚀 Starting validation gauntlet for hypothesis: {hypothesis.get('hypothesis_statement', '')[:50]}...")

        hypothesis_id = hypothesis.get('hypothesis_id', f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')")
        start_time = datetime.now()

        try:
            # Initialize tracking
            self.validation_history[hypothesis_id] = {
                'hypothesis': hypothesis,
                'tier_results': {},
                'resource_usage': {},
                'decisions': []
            }

            # Execute validation tiers sequentially
            tier_results = {}
            current_tier = ValidationTier.TIER_1
            tiers_completed = []
            total_cost = 0.0
            total_time = 0.0

            while current_tier:
                logger.info(f"📊 Executing {current_tier.value} validation...")

                # Execute current tier
                tier_start = datetime.now()
                tier_result = self._execute_tier_validation(current_tier, hypothesis, tier_results)
                tier_end = datetime.now()

                # Track resource usage
                tier_cost = self._calculate_tier_cost(current_tier, tier_result)
                tier_time = (tier_end - tier_start).total_seconds()

                total_cost += tier_cost
                total_time += tier_time

                # Store tier result
                tier_results[current_tier] = tier_result
                tiers_completed.append(current_tier)

                # Update tracking
                self.validation_history[hypothesis_id]['tier_results'][current_tier.value] = {
                    'result': tier_result,
                    'cost': tier_cost,
                    'time': tier_time,
                    'timestamp': tier_end.isoformat()
                }

                # Check if we should continue to next tier
                next_tier_decision = self._decide_next_tier(current_tier, tier_result, total_cost)

                if next_tier_decision['should_continue']:
                    current_tier = next_tier_decision['next_tier']
                    self.validation_history[hypothesis_id]['decisions'].append({
                        'tier': current_tier.value,
                        'decision': 'continue',
                        'reason': next_tier_decision['reason'],
                        'confidence': tier_result.confidence_score
                    })
                else:
                    # Stop validation process
                    self.validation_history[hypothesis_id]['decisions'].append({
                        'tier': current_tier.value,
                        'decision': 'stop',
                        'reason': next_tier_decision['reason'],
                        'confidence': tier_result.confidence_score
                    })
                    break

            # Calculate final results
            final_tier = current_tier or max(tiers_completed) if tiers_completed else ValidationTier.TIER_1
            overall_status = self._determine_overall_status(tier_results, final_tier)
            investment_readiness = self._calculate_investment_readiness(tier_results)
            final_recommendations = self._generate_final_recommendations(tier_results, overall_status)

            # Create comprehensive result
            result = ValidationGauntletResult(
                hypothesis_id=hypothesis_id,
                overall_status=overall_status,
                tiers_completed=tiers_completed,
                final_tier_reached=final_tier,
                validation_results=tier_results,
                resource_cost_total=round(total_cost, 2),
                execution_time_total=round(total_time, 2),
                investment_readiness_score=round(investment_readiness, 2),
                final_recommendations=final_recommendations,
                execution_timestamp=datetime.now(),
                performance_metrics=self._calculate_performance_metrics(tier_results, total_time, total_cost)
            )

            # Store comprehensive results
            self._store_gauntlet_results(result)

            logger.info(f"✅ Validation gauntlet completed for hypothesis {hypothesis_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Validation gauntlet failed: {e}")
            return ValidationGauntletResult(
                hypothesis_id=hypothesis_id,
                overall_status=ValidationStatus.FAILED,
                tiers_completed=[],
                final_tier_reached=ValidationTier.TIER_1,
                validation_results={},
                resource_cost_total=0.0,
                execution_time_total=(datetime.now() - start_time).total_seconds(),
                investment_readiness_score=0.0,
                final_recommendations=[f"Validation failed due to error: {str(e)}"],
                execution_timestamp=datetime.now()
            )

    def _execute_tier_validation(self, tier: ValidationTier, hypothesis: Dict[str, Any],
                               previous_results: Dict[ValidationTier, ValidationResult]) -> ValidationResult:
        """Execute validation for a specific tier"""
        try:
            if tier == ValidationTier.TIER_1:
                return self._execute_tier1_validation(hypothesis)
            elif tier == ValidationTier.TIER_2:
                sentiment_result = previous_results.get(ValidationTier.TIER_1)
                return self._execute_tier2_validation(hypothesis, sentiment_result)
            elif tier == ValidationTier.TIER_3:
                market_result = previous_results.get(ValidationTier.TIER_2)
                return self._execute_tier3_validation(hypothesis, market_result)
            elif tier == ValidationTier.TIER_4:
                prototype_result = previous_results.get(ValidationTier.TIER_3)
                return self._execute_tier4_validation(hypothesis, prototype_result)
            else:
                raise ValueError(f"Unknown validation tier: {tier}")

        except Exception as e:
            logger.error(f"❌ Tier {tier.value} validation failed: {e}")
            return ValidationResult(
                validation_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                hypothesis_id=hypothesis.get('hypothesis_id', ''),
                tier=tier,
                status=ValidationStatus.FAILED,
                confidence_score=0.0,
                evidence_sources=[],
                recommendations=[f"Validation failed: {str(e)}"],
                resource_cost=0.0,
                execution_time=0.0,
                next_tier_recommended=None,
                validation_data={'error': str(e)},
                timestamp=datetime.now()
            )

    def _execute_tier1_validation(self, hypothesis: Dict[str, Any]) -> ValidationResult:
        """Execute Tier 1: Sentiment Analysis"""
        sentiment_analysis = self.tier1_agent.analyze_sentiment(hypothesis)

        # Determine status based on sentiment
        if sentiment_analysis.sentiment_score >= 0.3:
            status = ValidationStatus.PASSED
            next_tier = ValidationTier.TIER_2
        elif sentiment_analysis.sentiment_score >= -0.1:
            status = ValidationStatus.PASSED  # Neutral but proceed
            next_tier = ValidationTier.TIER_2
        else:
            status = ValidationStatus.FAILED
            next_tier = None

        return ValidationResult(
            validation_id=sentiment_analysis.analysis_id,
            hypothesis_id=hypothesis.get('hypothesis_id', ''),
            tier=ValidationTier.TIER_1,
            status=status,
            confidence_score=abs(sentiment_analysis.sentiment_score),
            evidence_sources=['market_sentiment', 'social_signals', 'competitor_analysis'],
            recommendations=self._generate_tier1_recommendations(sentiment_analysis),
            resource_cost=50.0,  # Low cost for sentiment analysis
            execution_time=30.0,  # 30 seconds estimated
            next_tier_recommended=next_tier,
            validation_data={
                'sentiment_analysis': {
                    'overall_sentiment': sentiment_analysis.overall_sentiment,
                    'sentiment_score': sentiment_analysis.sentiment_score,
                    'market_receptivity_score': sentiment_analysis.market_receptivity_score
                }
            },
            timestamp=datetime.now()
        )

    def _execute_tier2_validation(self, hypothesis: Dict[str, Any], tier1_result: Optional[ValidationResult]) -> ValidationResult:
        """Execute Tier 2: Market Research Validation"""
        # Extract sentiment data from tier 1 if available
        sentiment_data = None
        if tier1_result and hasattr(tier1_result, 'validation_data'):
            sentiment_data = tier1_result.validation_data.get('sentiment_analysis')

        # Create mock sentiment analysis object for tier 2 agent
        mock_sentiment = None
        if sentiment_data:
            mock_sentiment = SentimentAnalysis(
                analysis_id="tier1_" + hypothesis.get('hypothesis_id', ''),
                hypothesis_id=hypothesis.get('hypothesis_id', ''),
                overall_sentiment=sentiment_data.get('overall_sentiment', 'neutral'),
                sentiment_score=sentiment_data.get('sentiment_score', 0.0),
                key_positive_signals=[],
                key_negative_signals=[],
                market_receptivity_score=sentiment_data.get('market_receptivity_score', 0.5),
                competitor_sentiment={},
                social_media_mentions=0,
                news_coverage_sentiment='neutral',
                analysis_timestamp=datetime.now()
            )

        market_validation = self.tier2_agent.validate_market_hypothesis(hypothesis, mock_sentiment)

        # Determine status based on market validation
        tech_feasibility = market_validation.technology_feasibility_score
        market_barriers = len(market_validation.go_to_market_barriers)

        if tech_feasibility >= 0.7 and market_barriers <= 2:
            status = ValidationStatus.PASSED
            next_tier = ValidationTier.TIER_3
        elif tech_feasibility >= 0.5:
            status = ValidationStatus.PASSED  # Proceed with caution
            next_tier = ValidationTier.TIER_3
        else:
            status = ValidationStatus.FAILED
            next_tier = None

        return ValidationResult(
            validation_id=market_validation.validation_id,
            hypothesis_id=hypothesis.get('hypothesis_id', ''),
            tier=ValidationTier.TIER_2,
            status=status,
            confidence_score=tech_feasibility,
            evidence_sources=['market_size_analysis', 'competitor_research', 'regulatory_review'],
            recommendations=self._generate_tier2_recommendations(market_validation),
            resource_cost=200.0,  # Medium cost for market research
            execution_time=120.0,  # 2 minutes estimated
            next_tier_recommended=next_tier,
            validation_data={
                'market_validation': {
                    'technology_feasibility': tech_feasibility,
                    'market_barriers_count': market_barriers,
                    'regulatory_considerations': market_validation.regulatory_considerations
                }
            },
            timestamp=datetime.now()
        )

    def _execute_tier3_validation(self, hypothesis: Dict[str, Any], tier2_result: Optional[ValidationResult]) -> ValidationResult:
        """Execute Tier 3: Prototype Generation and Testing"""
        # Extract market data from tier 2 if available
        market_data = None
        if tier2_result and hasattr(tier2_result, 'validation_data'):
            market_data = tier2_result.validation_data.get('market_validation')

        # Create mock market validation object for tier 3 agent
        mock_market = None
        if market_data:
            mock_market = MarketValidation(
                validation_id="tier2_" + hypothesis.get('hypothesis_id', ''),
                hypothesis_id=hypothesis.get('hypothesis_id', ''),
                market_size_validation={},
                competitor_analysis={},
                customer_segment_validation={},
                pricing_sensitivity_analysis={},
                regulatory_considerations=market_data.get('regulatory_considerations', []),
                technology_feasibility_score=market_data.get('technology_feasibility', 0.7),
                go_to_market_barriers=[],
                validation_timestamp=datetime.now()
            )

        prototype_result = self.tier3_agent.generate_and_test_prototype(hypothesis, mock_market)

        # Determine status based on prototype results
        usability_score = prototype_result.usability_score
        feature_validation = len(prototype_result.feature_validation_results)

        if usability_score >= 0.8 and feature_validation >= 3:
            status = ValidationStatus.PASSED
            next_tier = ValidationTier.TIER_4
        elif usability_score >= 0.6:
            status = ValidationStatus.PASSED  # Proceed with refinements needed
            next_tier = ValidationTier.TIER_4
        else:
            status = ValidationStatus.REQUIRES_REFINEMENT
            next_tier = None

        return ValidationResult(
            validation_id=prototype_result.prototype_id,
            hypothesis_id=hypothesis.get('hypothesis_id', ''),
            tier=ValidationTier.TIER_3,
            status=status,
            confidence_score=usability_score,
            evidence_sources=['user_testing', 'usability_metrics', 'feature_validation'],
            recommendations=prototype_result.iteration_recommendations,
            resource_cost=prototype_result.estimated_development_cost,
            execution_time=300.0,  # 5 minutes estimated
            next_tier_recommended=next_tier,
            validation_data={
                'prototype_result': {
                    'usability_score': usability_score,
                    'features_validated': feature_validation,
                    'development_complexity': prototype_result.development_complexity
                }
            },
            timestamp=datetime.now()
        )

    def _execute_tier4_validation(self, hypothesis: Dict[str, Any], tier3_result: Optional[ValidationResult]) -> ValidationResult:
        """Execute Tier 4: Interactive Prototype Validation"""
        # Extract prototype data from tier 3 if available
        prototype_data = None
        if tier3_result and hasattr(tier3_result, 'validation_data'):
            prototype_data = tier3_result.validation_data.get('prototype_result')

        # Create mock prototype result object for tier 4 agent
        mock_prototype = None
        if prototype_data:
            mock_prototype = PrototypeResult(
                prototype_id="tier3_" + hypothesis.get('hypothesis_id', ''),
                hypothesis_id=hypothesis.get('hypothesis_id', ''),
                prototype_type='interactive',
                prototype_url='',
                user_feedback_summary={},
                usability_score=prototype_data.get('usability_score', 0.7),
                feature_validation_results={},
                iteration_recommendations=[],
                development_complexity=prototype_data.get('development_complexity', 'medium'),
                estimated_development_cost=25000,
                prototype_timestamp=datetime.now()
            )

        interactive_validation = self.tier4_agent.conduct_interactive_validation(hypothesis, mock_prototype)

        # Determine final status based on investment readiness
        investment_readiness = interactive_validation.investment_readiness_score

        if investment_readiness >= 75:
            status = ValidationStatus.PASSED
        elif investment_readiness >= 60:
            status = ValidationStatus.PASSED  # Proceed with additional due diligence
        else:
            status = ValidationStatus.REQUIRES_REFINEMENT

        return ValidationResult(
            validation_id=interactive_validation.validation_id,
            hypothesis_id=hypothesis.get('hypothesis_id', ''),
            tier=ValidationTier.TIER_4,
            status=status,
            confidence_score=investment_readiness / 100.0,  # Convert to 0-1 scale
            evidence_sources=['user_testing', 'conversion_analysis', 'scalability_assessment'],
            recommendations=interactive_validation.final_recommendations,
            resource_cost=500.0,  # High cost for comprehensive validation
            execution_time=600.0,  # 10 minutes estimated
            next_tier_recommended=None,  # Final tier
            validation_data={
                'interactive_validation': {
                    'investment_readiness': investment_readiness,
                    'user_satisfaction': interactive_validation.user_testing_results.get('usability_metrics', {}).get('user_satisfaction', 0),
                    'scalability_score': interactive_validation.scalability_assessment.get('scalability_score', 0)
                }
            },
            timestamp=datetime.now()
        )

    def _decide_next_tier(self, current_tier: ValidationTier, tier_result: ValidationResult, total_cost: float) -> Dict[str, Any]:
        """Decide whether to continue to the next tier"""
        # Check if validation passed
        if tier_result.status == ValidationStatus.FAILED:
            if self.stop_on_failure:
                return {
                    'should_continue': False,
                    'reason': f"Validation failed at {current_tier.value}",
                    'next_tier': None
                }
            else:
                # Allow continuation with reduced confidence
                pass

        # Check budget constraints
        if total_cost + self._estimate_next_tier_cost(current_tier) > self.max_tier_budget:
            return {
                'should_continue': False,
                'reason': f"Budget limit exceeded (current: ${total_cost:.2f}, limit: ${self.max_tier_budget:.2f})",
                'next_tier': None
            }

        # Determine next tier
        tier_order = [ValidationTier.TIER_1, ValidationTier.TIER_2, ValidationTier.TIER_3, ValidationTier.TIER_4]
        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                next_tier = tier_order[current_index + 1]
                return {
                    'should_continue': True,
                    'reason': f"Proceeding to {next_tier.value} based on {tier_result.status.value} validation",
                    'next_tier': next_tier
                }
            else:
                return {
                    'should_continue': False,
                    'reason': "Reached final validation tier",
                    'next_tier': None
                }
        except ValueError:
            return {
                'should_continue': False,
                'reason': f"Unknown tier: {current_tier}",
                'next_tier': None
            }

    def _estimate_next_tier_cost(self, current_tier: ValidationTier) -> float:
        """Estimate cost of next tier validation"""
        cost_estimates = {
            ValidationTier.TIER_1: 50.0,
            ValidationTier.TIER_2: 200.0,
            ValidationTier.TIER_3: 500.0,
            ValidationTier.TIER_4: 1000.0
        }
        return cost_estimates.get(current_tier, 0.0)

    def _calculate_tier_cost(self, tier: ValidationTier, tier_result: ValidationResult) -> float:
        """Calculate actual cost for a completed tier"""
        return tier_result.resource_cost

    def _determine_overall_status(self, tier_results: Dict[ValidationTier, ValidationResult],
                                final_tier: ValidationTier) -> ValidationStatus:
        """Determine overall validation status"""
        if not tier_results:
            return ValidationStatus.FAILED

        # If reached tier 4 and passed, overall success
        if final_tier == ValidationTier.TIER_4:
            tier4_result = tier_results.get(ValidationTier.TIER_4)
            if tier4_result and tier4_result.status == ValidationStatus.PASSED:
                return ValidationStatus.PASSED

        # Check if any tier failed critically
        for tier, result in tier_results.items():
            if result.status == ValidationStatus.FAILED:
                return ValidationStatus.FAILED

        # If reached at least tier 3, consider partial success
        if final_tier in [ValidationTier.TIER_3, ValidationTier.TIER_4]:
            return ValidationStatus.PASSED

        return ValidationStatus.REQUIRES_REFINEMENT

    def _calculate_investment_readiness(self, tier_results: Dict[ValidationTier, ValidationResult]) -> float:
        """Calculate overall investment readiness score"""
        if not tier_results:
            return 0.0

        # Base score from tier 4 if available
        if ValidationTier.TIER_4 in tier_results:
            tier4_result = tier_results[ValidationTier.TIER_4]
            if hasattr(tier4_result, 'validation_data'):
                interactive_data = tier4_result.validation_data.get('interactive_validation', {})
                return interactive_data.get('investment_readiness', 50.0)

        # Fallback scoring based on tiers completed
        tier_weights = {
            ValidationTier.TIER_1: 0.1,
            ValidationTier.TIER_2: 0.2,
            ValidationTier.TIER_3: 0.3,
            ValidationTier.TIER_4: 0.4
        }

        total_score = 0.0
        total_weight = 0.0

        for tier, result in tier_results.items():
            weight = tier_weights.get(tier, 0.1)
            confidence = result.confidence_score
            total_score += confidence * weight
            total_weight += weight

        return (total_score / total_weight) * 100 if total_weight > 0 else 0.0

    def _generate_final_recommendations(self, tier_results: Dict[ValidationTier, ValidationResult],
                                      overall_status: ValidationStatus) -> List[str]:
        """Generate final recommendations based on all validation results"""
        recommendations = []

        if overall_status == ValidationStatus.PASSED:
            recommendations.extend([
                "Proceed with product development and seek funding",
                "Strong validation results across all tiers",
                "High confidence in market opportunity and product-market fit"
            ])
        elif overall_status == ValidationStatus.REQUIRES_REFINEMENT:
            recommendations.extend([
                "Address identified issues before proceeding",
                "Consider additional user testing and market research",
                "Re-evaluate business model assumptions"
            ])
        else:
            recommendations.extend([
                "Reconsider market opportunity or pivot strategy",
                "Validation indicates significant risks or challenges",
                "Consider alternative approaches or market segments"
            ])

        # Add tier-specific recommendations
        for tier, result in tier_results.items():
            if result.recommendations:
                recommendations.extend([f"{tier.value.upper()}: {rec}" for rec in result.recommendations[:2]])

        return recommendations[:10]  # Limit to top 10 recommendations

    def _calculate_performance_metrics(self, tier_results: Dict[ValidationTier, ValidationResult],
                                     total_time: float, total_cost: float) -> Dict[str, Any]:
        """Calculate performance metrics for the validation process"""
        return {
            'tiers_completed': len(tier_results),
            'average_confidence': sum(r.confidence_score for r in tier_results.values()) / len(tier_results) if tier_results else 0,
            'total_execution_time': total_time,
            'total_cost': total_cost,
            'cost_per_tier': total_cost / len(tier_results) if tier_results else 0,
            'time_per_tier': total_time / len(tier_results) if tier_results else 0,
            'validation_efficiency': len(tier_results) / total_time if total_time > 0 else 0
        }

    def _generate_tier1_recommendations(self, sentiment_analysis: SentimentAnalysis) -> List[str]:
        """Generate recommendations based on tier 1 sentiment analysis"""
        recommendations = []

        if sentiment_analysis.sentiment_score < 0:
            recommendations.append("Address negative sentiment through messaging improvements")
        if sentiment_analysis.market_receptivity_score < 0.5:
            recommendations.append("Enhance market positioning and value proposition")
        if sentiment_analysis.social_media_mentions < 10:
            recommendations.append("Increase market awareness and community engagement")

        return recommendations or ["Proceed to market research validation"]

    def _generate_tier2_recommendations(self, market_validation: MarketValidation) -> List[str]:
        """Generate recommendations based on tier 2 market validation"""
        recommendations = []

        if market_validation.technology_feasibility_score < 0.6:
            recommendations.append("Address technical feasibility concerns")
        if len(market_validation.regulatory_considerations) > 3:
            recommendations.append("Conduct detailed regulatory compliance review")
        if len(market_validation.go_to_market_barriers) > 2:
            recommendations.append("Develop strategies to overcome market entry barriers")

        return recommendations or ["Proceed to prototype development and testing"]

    def _store_gauntlet_results(self, result: ValidationGauntletResult) -> bool:
        """Store complete validation gauntlet results in Supabase"""
        if not self.supabase:
            logger.warning("Supabase unavailable - gauntlet results not stored")
            return False

        try:
            storage_data = {
                'analysis_type': 'validation_gauntlet',
                'gauntlet_result': {
                    'hypothesis_id': result.hypothesis_id,
                    'overall_status': result.overall_status.value,
                    'tiers_completed': [tier.value for tier in result.tiers_completed],
                    'final_tier_reached': result.final_tier_reached.value,
                    'resource_cost_total': result.resource_cost_total,
                    'execution_time_total': result.execution_time_total,
                    'investment_readiness_score': result.investment_readiness_score,
                    'final_recommendations': result.final_recommendations,
                    'performance_metrics': result.performance_metrics
                },
                'validation_results': {
                    tier.value: {
                        'status': result.validation_results[tier].status.value,
                        'confidence_score': result.validation_results[tier].confidence_score,
                        'recommendations': result.validation_results[tier].recommendations
                    } for tier in result.validation_results.keys()
                },
                'timestamp': result.execution_timestamp.isoformat(),
                'source': 'validation_gauntlet_orchestrator'
            }

            db_result = self.supabase.table('market_intelligence').insert(storage_data).execute()

            if db_result.data:
                logger.info("✅ Validation gauntlet results stored successfully")
                return True
            else:
                logger.error("❌ Failed to store validation gauntlet results")
                return False

        except Exception as e:
            logger.error(f"Error storing validation gauntlet results: {e}")
            return False

def main():
    """Main execution function for validation gauntlet"""
    try:
        # Initialize orchestrator
        orchestrator = ValidationGauntletOrchestrator()

        # Example hypothesis for testing
        test_hypothesis = {
            'hypothesis_id': f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'hypothesis_statement': 'We believe that small business owners need an AI-powered workflow automation platform that can reduce manual processes by 60% and improve operational efficiency.',
            'market_size_estimate': '$5B globally',
            'target_market': 'SMB technology solutions',
            'key_assumptions': [
                'Small businesses struggle with manual processes',
                'AI technology can significantly improve efficiency',
                'Businesses are willing to pay for automation solutions'
            ]
        }

        # Execute validation gauntlet
        logger.info("🎯 Starting validation gauntlet execution...")
        result = orchestrator.execute_validation_gauntlet(test_hypothesis)

        # Display results
        print(f"\n🎯 VALIDATION GAUNTLET RESULTS")
        print(f"Hypothesis ID: {result.hypothesis_id}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Final Tier Reached: {result.final_tier_reached.value}")
        print(f"Tiers Completed: {[tier.value for tier in result.tiers_completed]}")
        print(f"Investment Readiness Score: {result.investment_readiness_score}%")
        print(f"Total Cost: ${result.resource_cost_total}")
        print(f"Total Time: {result.execution_time_total:.1f} seconds")

        print(f"\n📋 FINAL RECOMMENDATIONS:")
        for i, rec in enumerate(result.final_recommendations, 1):
            print(f"{i}. {rec}")

        print(f"\n✅ Validation gauntlet completed successfully!")

    except Exception as e:
        logger.error(f"❌ Validation gauntlet execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()