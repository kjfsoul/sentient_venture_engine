#!/usr/bin/env python3
"""
Causal Analysis Methods for SVE Project
Contains feature extraction and causal inference methods for the CausalAnalysisAgent
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class CausalAnalysisMethods:
    """Methods for feature extraction and causal analysis"""
    
    @staticmethod
    def extract_market_complexity(hypothesis: Dict, metrics: Dict) -> float:
        """Extract market complexity score from hypothesis and metrics"""
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        complexity_indicators = ['enterprise', 'b2b', 'platform', 'integration', 'ai', 'ml', 'saas', 'api']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text)
        return min(complexity_score / len(complexity_indicators), 1.0)
    
    @staticmethod
    def extract_validation_strategy(record: Dict) -> str:
        """Extract validation strategy type"""
        tier = record.get('tier', 1)
        strategy_map = {
            1: 'social_sentiment',
            2: 'prototype_testing', 
            3: 'market_validation'
        }
        return strategy_map.get(tier, 'unknown')
    
    @staticmethod
    def extract_resource_investment(metrics: Dict) -> float:
        """Extract resource investment level"""
        if not metrics:
            return 0.3
        
        # Calculate based on metrics complexity and values
        investment_indicators = ['conversion_rate', 'user_engagement', 'retention_rate', 'cost_per_acquisition']
        present_metrics = sum(1 for indicator in investment_indicators if indicator in metrics)
        
        # Higher investment if more comprehensive metrics are tracked
        base_investment = min(present_metrics / len(investment_indicators), 1.0)
        
        # Adjust based on metric values (higher values suggest more investment)
        metric_values = [v for k, v in metrics.items() if isinstance(v, (int, float)) and v > 0]
        if metric_values:
            avg_metric = np.mean(metric_values)
            investment_multiplier = min(avg_metric, 1.0) if avg_metric < 10 else min(avg_metric / 100, 1.0)
            return min(base_investment * (1 + investment_multiplier), 1.0)
        
        return base_investment
    
    @staticmethod
    def extract_hypothesis_novelty(hypothesis: Dict) -> float:
        """Extract hypothesis novelty score"""
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        novelty_indicators = ['new', 'novel', 'innovative', 'first', 'breakthrough', 'revolutionary', 'unique', 'disruptive']
        incremental_indicators = ['improve', 'enhance', 'optimize', 'better', 'faster', 'cheaper']
        
        novelty_score = sum(1 for indicator in novelty_indicators if indicator in text)
        incremental_score = sum(1 for indicator in incremental_indicators if indicator in text)
        
        # Balance novelty vs incremental indicators
        if novelty_score > incremental_score:
            return min(novelty_score / len(novelty_indicators), 1.0)
        elif incremental_score > novelty_score:
            return max(0.2, 1.0 - (incremental_score / len(incremental_indicators)))
        else:
            return 0.5  # Neutral
    
    @staticmethod
    def extract_market_timing(record: Dict) -> float:
        """Extract market timing score"""
        # Analyze validation timestamp vs current market conditions
        validation_time = record.get('validation_timestamp')
        if not validation_time:
            return 0.5
        
        try:
            val_date = datetime.fromisoformat(validation_time.replace('Z', '+00:00'))
            current_date = datetime.now()
            days_ago = (current_date - val_date).days
            
            # Recent validations get higher timing scores (more current market conditions)
            if days_ago <= 7:
                return 0.9
            elif days_ago <= 30:
                return 0.7
            elif days_ago <= 90:
                return 0.5
            else:
                return 0.3
        except:
            return 0.5
    
    @staticmethod
    def extract_user_engagement(metrics: Dict) -> float:
        """Extract user engagement score"""
        engagement_metrics = ['user_engagement', 'interaction_rate', 'retention_rate', 'session_duration', 'click_through_rate']
        scores = []
        
        for metric in engagement_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    # Normalize different metric types
                    if metric in ['retention_rate', 'click_through_rate']:
                        scores.append(min(value, 1.0))  # Assume percentage
                    elif metric == 'session_duration':
                        scores.append(min(value / 300, 1.0))  # Normalize to 5 minutes max
                    else:
                        scores.append(min(value, 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    @staticmethod
    def extract_feedback_quality(feedback: List[Dict]) -> float:
        """Extract feedback quality score"""
        if not feedback:
            return 0.3
        
        quality_scores = []
        for f in feedback:
            rationale = f.get('rationale_text', '')
            decision = f.get('human_decision', '')
            
            if rationale:
                # Quality based on rationale length and content richness
                words = rationale.split()
                word_count_score = min(len(words) / 50.0, 1.0)
                
                # Bonus for specific decision rationale
                decision_clarity = 0.2 if decision in ['approve', 'reject', 'modify'] else 0.0
                
                quality_score = word_count_score + decision_clarity
                quality_scores.append(min(quality_score, 1.0))
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    @staticmethod
    def extract_iteration_speed(record: Dict) -> float:
        """Extract iteration speed metric"""
        # Based on validation tier progression
        tier = record.get('tier', 1)
        tier_progress = record.get('validation_tier_progress', 0)
        
        # Higher tier with good progress indicates faster iteration
        speed_score = (tier * 0.3) + (tier_progress * 0.7)
        return min(speed_score / 3.0, 1.0)  # Normalize to max tier 3
    
    @staticmethod
    def extract_market_conditions(record: Dict) -> float:
        """Extract market conditions confounding variable"""
        # Based on validation success patterns and timing
        validation_time = record.get('validation_timestamp')
        pass_fail = record.get('pass_fail_status', 'fail')
        
        # Recent successful validations suggest better market conditions
        if validation_time:
            try:
                val_date = datetime.fromisoformat(validation_time.replace('Z', '+00:00'))
                days_ago = (datetime.now() - val_date).days
                
                base_conditions = 0.7 if days_ago <= 30 else 0.5
                success_bonus = 0.2 if pass_fail == 'pass' else -0.1
                
                return max(0.1, min(base_conditions + success_bonus, 1.0))
            except:
                pass
        
        return 0.5
    
    @staticmethod
    def extract_team_experience(hypothesis: Dict) -> float:
        """Extract team experience confounding variable"""
        agent = hypothesis.get('generated_by_agent', '').lower()
        
        # Experience mapping based on agent sophistication
        experience_map = {
            'synthesis_agent': 0.8,
            'market_intel_agent': 0.7,
            'multimodal_agent': 0.6,
            'validation_agent': 0.7,
            'causal_analysis_agent': 0.9
        }
        
        for agent_type, score in experience_map.items():
            if agent_type in agent:
                return score
        
        # Default based on hypothesis complexity
        text = hypothesis.get('initial_hypothesis_text', '')
        complexity_score = len(text.split()) / 100.0  # Longer hypotheses suggest more experience
        return min(0.5 + complexity_score, 1.0)
    
    @staticmethod
    def extract_competitive_landscape(metrics: Dict) -> float:
        """Extract competitive landscape confounding variable"""
        # Infer from market-related metrics
        competitive_indicators = ['market_share', 'competitive_advantage', 'differentiation_score']
        competition_scores = []
        
        for indicator in competitive_indicators:
            if indicator in metrics:
                value = metrics[indicator]
                if isinstance(value, (int, float)):
                    competition_scores.append(min(value, 1.0))
        
        if competition_scores:
            return np.mean(competition_scores)
        
        # Fallback: infer from other metrics
        if metrics:
            # More metrics tracked might indicate more competitive environment
            metric_count = len([v for v in metrics.values() if isinstance(v, (int, float))])
            return min(metric_count / 10.0, 1.0)
        
        return 0.5
    
    @staticmethod
    def extract_time_to_validation(record: Dict) -> float:
        """Extract time to validation outcome"""
        # Calculate based on tier and timestamp
        tier = record.get('tier', 1)
        validation_time = record.get('validation_timestamp')
        
        if validation_time:
            try:
                val_date = datetime.fromisoformat(validation_time.replace('Z', '+00:00'))
                # Assume hypothesis creation was some days before validation
                estimated_days = tier * 3 + np.random.uniform(1, 7)  # 4-10 days for tier 1, etc.
                return estimated_days
            except:
                pass
        
        # Fallback estimation
        return tier * 5 + np.random.uniform(1, 10)
    
    @staticmethod
    def extract_cost_efficiency(metrics: Dict) -> float:
        """Extract cost efficiency outcome"""
        # Based on resource utilization metrics
        efficiency_indicators = ['cost_per_acquisition', 'roi', 'conversion_rate', 'resource_utilization']
        efficiency_scores = []
        
        for indicator in efficiency_indicators:
            if indicator in metrics:
                value = metrics[indicator]
                if isinstance(value, (int, float)) and value > 0:
                    if indicator == 'cost_per_acquisition':
                        # Lower cost per acquisition = higher efficiency
                        efficiency_scores.append(max(0.1, 1.0 - min(value / 100, 1.0)))
                    elif indicator in ['roi', 'conversion_rate']:
                        efficiency_scores.append(min(value, 1.0))
                    else:
                        efficiency_scores.append(min(value, 1.0))
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.5
    
    @staticmethod
    def generate_simulated_data(n_samples: int = 100) -> pd.DataFrame:
        """Generate simulated data for testing causal analysis"""
        logger.info(f"🧪 Generating {n_samples} simulated validation records for causal analysis")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate simulated data with realistic causal relationships
        data = {
            'hypothesis_id': [f"hyp_{i:03d}" for i in range(n_samples)],
            'validation_tier': np.random.choice([1, 2, 3], n_samples),
            'market_complexity': np.random.uniform(0, 1, n_samples),
            'validation_strategy': np.random.choice(['social_sentiment', 'prototype_testing', 'market_validation'], n_samples),
            'resource_investment': np.random.uniform(0, 1, n_samples),
            'hypothesis_novelty': np.random.uniform(0, 1, n_samples),
            'market_timing': np.random.uniform(0, 1, n_samples),
            'user_engagement': np.random.uniform(0, 1, n_samples),
            'feedback_quality': np.random.uniform(0, 1, n_samples),
            'iteration_speed': np.random.uniform(0, 1, n_samples),
            'market_conditions': np.random.uniform(0, 1, n_samples),
            'team_experience': np.random.uniform(0, 1, n_samples),
            'competitive_landscape': np.random.uniform(0, 1, n_samples),
        }
        
        # Generate outcomes with realistic causal relationships
        validation_success_prob = (
            0.3 * data['resource_investment'] +
            0.2 * data['user_engagement'] +
            0.2 * data['team_experience'] +
            0.1 * (1 - data['market_complexity']) +
            0.2 * data['market_conditions']
        )
        data['validation_success'] = np.random.binomial(1, validation_success_prob)
        
        data['time_to_validation'] = (
            10 + 5 * data['market_complexity'] +
            5 * (1 - data['team_experience']) +
            np.random.normal(0, 2, n_samples)
        )
        data['time_to_validation'] = np.maximum(data['time_to_validation'], 1)  # Minimum 1 day
        
        data['cost_efficiency'] = (
            data['team_experience'] * 0.4 +
            data['resource_investment'] * 0.3 +
            (1 - data['market_complexity']) * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['cost_efficiency'] = np.clip(data['cost_efficiency'], 0, 1)
        
        data['human_approval'] = np.random.binomial(
            1, 
            0.7 * data['validation_success'] + 0.3 * data['user_engagement']
        )
        
        return pd.DataFrame(data)