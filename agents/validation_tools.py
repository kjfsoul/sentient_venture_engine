#!/usr/bin/env python3
"""
Phase 3: Validation Tools and Utilities
Specialized utilities supporting the Tiered Validation Gauntlet

Provides:
- Validation metrics calculators
- Data processing utilities
- Statistical analysis tools
- Sentiment analysis helpers
- Prototype generation utilities
- User testing frameworks
"""

import os
import sys
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Container for validation performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime

class ValidationMetricsCalculator:
    """Calculator for validation performance metrics"""

    @staticmethod
    def calculate_binary_classification_metrics(predictions: List[int], actuals: List[int]) -> ValidationMetrics:
        """Calculate binary classification metrics"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        true_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        false_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
        true_negatives = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
        false_negatives = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(predictions)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate confidence interval (simplified)
        confidence_interval = ValidationMetricsCalculator._calculate_confidence_interval(accuracy, len(predictions))

        return ValidationMetrics(
            accuracy=round(accuracy, 3),
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1_score=round(f1_score, 3),
            confidence_interval=confidence_interval,
            sample_size=len(predictions),
            timestamp=datetime.now()
        )

    @staticmethod
    def _calculate_confidence_interval(accuracy: float, sample_size: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for accuracy"""
        # Simplified calculation using normal approximation
        standard_error = (accuracy * (1 - accuracy) / sample_size) ** 0.5
        z_score = 1.96  # For 95% confidence
        margin_of_error = z_score * standard_error

        lower_bound = max(0.0, accuracy - margin_of_error)
        upper_bound = min(1.0, accuracy + margin_of_error)

        return (round(lower_bound, 3), round(upper_bound, 3))

    @staticmethod
    def calculate_sentiment_metrics(sentiment_scores: List[float], actual_sentiments: List[str]) -> Dict[str, Any]:
        """Calculate sentiment analysis performance metrics"""
        # Convert actual sentiments to numerical scores
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        numerical_actuals = [sentiment_mapping.get(s.lower(), 0) for s in actual_sentiments]

        # Calculate correlation
        if len(sentiment_scores) > 1:
            correlation = ValidationMetricsCalculator._calculate_correlation(sentiment_scores, numerical_actuals)
        else:
            correlation = 0.0

        # Calculate accuracy within tolerance
        tolerance = 0.3
        accurate_predictions = sum(1 for pred, actual in zip(sentiment_scores, numerical_actuals)
                                 if abs(pred - actual) <= tolerance)

        accuracy = accurate_predictions / len(sentiment_scores) if sentiment_scores else 0

        return {
            'correlation_coefficient': round(correlation, 3),
            'accuracy_within_tolerance': round(accuracy, 3),
            'mean_absolute_error': round(statistics.mean([abs(p - a) for p, a in zip(sentiment_scores, numerical_actuals)]), 3),
            'root_mean_squared_error': round((statistics.mean([(p - a)**2 for p, a in zip(sentiment_scores, numerical_actuals)]))**0.5, 3),
            'sentiment_distribution': ValidationMetricsCalculator._analyze_sentiment_distribution(sentiment_scores)
        }

    @staticmethod
    def _calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        try:
            return statistics.correlation(x, y)
        except:
            # Fallback calculation
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi**2 for xi in x)
            sum_y2 = sum(yi**2 for yi in y)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5

            return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def _analyze_sentiment_distribution(sentiment_scores: List[float]) -> Dict[str, Any]:
        """Analyze distribution of sentiment scores"""
        if not sentiment_scores:
            return {'negative': 0, 'neutral': 0, 'positive': 0}

        negative = sum(1 for s in sentiment_scores if s < -0.3)
        neutral = sum(1 for s in sentiment_scores if -0.3 <= s <= 0.3)
        positive = sum(1 for s in sentiment_scores if s > 0.3)

        total = len(sentiment_scores)

        return {
            'negative': round(negative / total, 3),
            'neutral': round(neutral / total, 3),
            'positive': round(positive / total, 3),
            'mean_score': round(statistics.mean(sentiment_scores), 3),
            'std_deviation': round(statistics.stdev(sentiment_scores), 3) if len(sentiment_scores) > 1 else 0.0
        }

class SentimentAnalysisTools:
    """Tools for sentiment analysis and text processing"""

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove punctuation but keep sentence structure
        text = re.sub(r'[^\w\s.!?]', '', text)

        return text.strip()

    @staticmethod
    def extract_sentiment_keywords(text: str) -> Dict[str, List[str]]:
        """Extract sentiment-bearing keywords from text"""
        positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'awesome', 'brilliant', 'outstanding', 'superb',
            'impressive', 'remarkable', 'exceptional', 'perfect', 'ideal'
        ]

        negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
            'poor', 'inadequate', 'disappointing', 'frustrating', 'annoying',
            'useless', 'broken', 'failed', 'problem', 'issue', 'error'
        ]

        neutral_keywords = [
            'okay', 'fine', 'average', 'normal', 'standard', 'typical',
            'moderate', 'reasonable', 'acceptable', 'satisfactory'
        ]

        text_lower = text.lower()

        positive_found = [word for word in positive_keywords if word in text_lower]
        negative_found = [word for word in negative_keywords if word in text_lower]
        neutral_found = [word for word in neutral_keywords if word in text_lower]

        return {
            'positive': positive_found,
            'negative': negative_found,
            'neutral': neutral_found,
            'sentiment_intensity': SentimentAnalysisTools._calculate_sentiment_intensity(
                positive_found, negative_found, neutral_found
            )
        }

    @staticmethod
    def _calculate_sentiment_intensity(positive: List[str], negative: List[str], neutral: List[str]) -> Dict[str, Any]:
        """Calculate sentiment intensity metrics"""
        total_words = len(positive) + len(negative) + len(neutral)

        if total_words == 0:
            return {'intensity_score': 0.0, 'dominant_sentiment': 'neutral'}

        positive_ratio = len(positive) / total_words
        negative_ratio = len(negative) / total_words
        neutral_ratio = len(neutral) / total_words

        # Calculate intensity score
        intensity_score = positive_ratio - negative_ratio

        # Determine dominant sentiment
        if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
            dominant = 'positive'
        elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
            dominant = 'negative'
        else:
            dominant = 'neutral'

        return {
            'intensity_score': round(intensity_score, 3),
            'dominant_sentiment': dominant,
            'sentiment_ratios': {
                'positive': round(positive_ratio, 3),
                'negative': round(negative_ratio, 3),
                'neutral': round(neutral_ratio, 3)
            }
        }

    @staticmethod
    def analyze_text_complexity(text: str) -> Dict[str, Any]:
        """Analyze text complexity for validation purposes"""
        if not text:
            return {'complexity_score': 0.0}

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        words = text.split()
        unique_words = set(words)

        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        vocabulary_richness = len(unique_words) / len(words) if words else 0

        # Flesch Reading Ease score (simplified)
        if words:
            syllables = sum(SentimentAnalysisTools._count_syllables(word) for word in words)
            flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (syllables / len(words))
            flesch_score = max(0, min(100, flesch_score))  # Clamp to valid range
        else:
            flesch_score = 0

        # Complexity classification
        if flesch_score >= 80:
            complexity_level = 'very_easy'
        elif flesch_score >= 60:
            complexity_level = 'easy'
        elif flesch_score >= 40:
            complexity_level = 'medium'
        elif flesch_score >= 20:
            complexity_level = 'difficult'
        else:
            complexity_level = 'very_difficult'

        return {
            'complexity_score': round(flesch_score, 1),
            'complexity_level': complexity_level,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'vocabulary_richness': round(vocabulary_richness, 3),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(unique_words)
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word (simplified algorithm)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word[0] in vowels:
            count += 1

        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        if count == 0:
            count += 1

        return count

class MarketAnalysisTools:
    """Tools for market analysis and validation"""

    @staticmethod
    def calculate_market_penetration_rate(market_size: Union[str, float], current_users: int) -> Dict[str, Any]:
        """Calculate market penetration rate"""
        try:
            # Parse market size if it's a string
            if isinstance(market_size, str):
                # Extract numerical value and unit
                market_size_num = float(re.findall(r'[\d,]+(?:\.\d+)?', market_size.replace(',', ''))[0])
                if 'billion' in market_size.lower() or 'B' in market_size:
                    market_size_num *= 1000000000
                elif 'million' in market_size.lower() or 'M' in market_size:
                    market_size_num *= 1000000
                elif 'thousand' in market_size.lower() or 'K' in market_size:
                    market_size_num *= 1000
            else:
                market_size_num = market_size

            penetration_rate = current_users / market_size_num if market_size_num > 0 else 0

            return {
                'penetration_rate': round(penetration_rate, 4),
                'penetration_percentage': round(penetration_rate * 100, 2),
                'market_size_numeric': market_size_num,
                'current_users': current_users,
                'market_opportunity': market_size_num - current_users
            }

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse market size: {market_size}")
            return {
                'penetration_rate': 0.0,
                'penetration_percentage': 0.0,
                'error': f"Could not parse market size: {str(e)}"
            }

    @staticmethod
    def analyze_competitive_positioning(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive positioning"""
        competitors = market_data.get('competitors', [])
        if not competitors:
            return {'positioning_score': 0.5, 'market_position': 'unknown'}

        # Calculate average market share
        market_shares = [comp.get('market_share', 0) for comp in competitors if isinstance(comp.get('market_share'), (int, float))]
        avg_market_share = statistics.mean(market_shares) if market_shares else 0

        # Determine positioning based on market share distribution
        if avg_market_share > 30:
            position = 'highly_concentrated'
            score = 0.8
        elif avg_market_share > 15:
            position = 'moderately_concentrated'
            score = 0.6
        elif avg_market_share > 5:
            position = 'fragmented'
            score = 0.4
        else:
            position = 'highly_fragmented'
            score = 0.2

        return {
            'positioning_score': score,
            'market_position': position,
            'average_market_share': round(avg_market_share, 2),
            'market_concentration_index': MarketAnalysisTools._calculate_hhi(market_shares),
            'competitive_intensity': MarketAnalysisTools._assess_competitive_intensity(market_shares)
        }

    @staticmethod
    def _calculate_hhi(market_shares: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index (HHI)"""
        if not market_shares:
            return 0.0

        # Normalize market shares to sum to 100
        total_share = sum(market_shares)
        normalized_shares = [share / total_share * 100 for share in market_shares] if total_share > 0 else []

        hhi = sum(share ** 2 for share in normalized_shares)

        return round(hhi, 2)

    @staticmethod
    def _assess_competitive_intensity(market_shares: List[float]) -> str:
        """Assess competitive intensity based on market shares"""
        hhi = MarketAnalysisTools._calculate_hhi(market_shares)

        if hhi > 2500:
            return 'low'  # Highly concentrated
        elif hhi > 1500:
            return 'moderate'
        else:
            return 'high'  # Highly competitive

    @staticmethod
    def calculate_customer_acquisition_cost(revenue: float, new_customers: int, marketing_spend: float) -> Dict[str, Any]:
        """Calculate Customer Acquisition Cost (CAC)"""
        if new_customers <= 0:
            return {'cac': 0.0, 'error': 'Invalid number of new customers'}

        cac = marketing_spend / new_customers
        estimated_clv = revenue / new_customers if new_customers > 0 else 0
        clv_cac_ratio = estimated_clv / cac if cac > 0 else 0

        return {
            'cac': round(cac, 2),
            'estimated_clv': round(estimated_clv, 2),
            'clv_cac_ratio': round(clv_cac_ratio, 2),
            'profitability_threshold': clv_cac_ratio > 3.0,
            'marketing_efficiency': 'good' if clv_cac_ratio > 3.0 else 'needs_improvement'
        }

class PrototypeGenerationTools:
    """Tools for prototype generation and testing"""

    @staticmethod
    def generate_user_flow_diagram(features: List[str], user_type: str = 'end_user') -> Dict[str, Any]:
        """Generate user flow diagram structure"""
        # Define standard user flows
        standard_flows = {
            'end_user': [
                'Landing Page',
                'Sign Up / Login',
                'Onboarding',
                'Main Dashboard',
                'Core Features',
                'Settings',
                'Help & Support'
            ],
            'admin': [
                'Admin Login',
                'Dashboard',
                'User Management',
                'Content Management',
                'Analytics',
                'System Settings'
            ],
            'api_user': [
                'API Documentation',
                'Authentication',
                'API Testing',
                'Rate Limiting',
                'Error Handling'
            ]
        }

        base_flow = standard_flows.get(user_type, standard_flows['end_user'])

        # Customize flow based on features
        customized_flow = PrototypeGenerationTools._customize_flow_for_features(base_flow, features)

        return {
            'user_type': user_type,
            'flow_steps': customized_flow,
            'total_steps': len(customized_flow),
            'estimated_completion_time': f"{len(customized_flow) * 2} minutes",
            'complexity_score': PrototypeGenerationTools._calculate_flow_complexity(customized_flow)
        }

    @staticmethod
    def _customize_flow_for_features(base_flow: List[str], features: List[str]) -> List[str]:
        """Customize user flow based on specific features"""
        customized_flow = base_flow.copy()

        # Add feature-specific steps
        feature_mappings = {
            'analytics': 'Analytics Dashboard',
            'reporting': 'Reports & Insights',
            'automation': 'Automation Setup',
            'integration': 'Integrations',
            'collaboration': 'Team Collaboration',
            'customization': 'Customization Options'
        }

        for feature in features:
            feature_lower = feature.lower()
            for key, step in feature_mappings.items():
                if key in feature_lower and step not in customized_flow:
                    # Insert after main dashboard
                    if 'Main Dashboard' in customized_flow:
                        insert_index = customized_flow.index('Main Dashboard') + 1
                        customized_flow.insert(insert_index, step)
                    else:
                        customized_flow.append(step)

        return customized_flow

    @staticmethod
    def _calculate_flow_complexity(flow_steps: List[str]) -> Dict[str, Any]:
        """Calculate complexity score for user flow"""
        complexity_score = len(flow_steps) * 0.1  # Base complexity

        # Add complexity for advanced features
        advanced_indicators = ['analytics', 'integration', 'automation', 'customization']
        advanced_steps = sum(1 for step in flow_steps if any(indicator in step.lower() for indicator in advanced_indicators))

        complexity_score += advanced_steps * 0.2

        if complexity_score >= 1.0:
            complexity_level = 'high'
        elif complexity_score >= 0.6:
            complexity_level = 'medium'
        else:
            complexity_level = 'low'

        return {
            'complexity_score': round(complexity_score, 2),
            'complexity_level': complexity_level,
            'advanced_features_count': advanced_steps
        }

    @staticmethod
    def generate_wireframe_specification(features: List[str]) -> Dict[str, Any]:
        """Generate wireframe specification"""
        screens = []

        # Generate main screens
        for feature in features:
            screen = {
                'name': f"{feature.title()} Screen",
                'purpose': f"Allow users to {feature.lower()}",
                'key_elements': PrototypeGenerationTools._generate_screen_elements(feature),
                'user_actions': PrototypeGenerationTools._generate_user_actions(feature)
            }
            screens.append(screen)

        return {
            'total_screens': len(screens),
            'screens': screens,
            'navigation_structure': 'tab-based' if len(screens) <= 5 else 'drawer-based',
            'responsive_breakpoints': ['mobile', 'tablet', 'desktop'],
            'estimated_fidelity': 'medium'
        }

    @staticmethod
    def _generate_screen_elements(feature: str) -> List[str]:
        """Generate UI elements for a feature screen"""
        base_elements = ['Header', 'Content Area', 'Action Buttons']

        feature_lower = feature.lower()

        if 'dashboard' in feature_lower:
            base_elements.extend(['Charts', 'Metrics Cards', 'Data Tables'])
        elif 'form' in feature_lower or 'input' in feature_lower:
            base_elements.extend(['Input Fields', 'Validation Messages', 'Submit Button'])
        elif 'list' in feature_lower or 'table' in feature_lower:
            base_elements.extend(['Search Bar', 'Filters', 'Sort Options', 'Pagination'])
        elif 'analytics' in feature_lower:
            base_elements.extend(['Graphs', 'KPIs', 'Date Range Picker'])

        return base_elements

    @staticmethod
    def _generate_user_actions(feature: str) -> List[str]:
        """Generate possible user actions for a feature"""
        actions = ['View', 'Navigate']

        feature_lower = feature.lower()

        if 'create' in feature_lower or 'add' in feature_lower:
            actions.extend(['Create New', 'Save'])
        if 'edit' in feature_lower or 'update' in feature_lower:
            actions.extend(['Edit', 'Update', 'Cancel'])
        if 'delete' in feature_lower or 'remove' in feature_lower:
            actions.extend(['Delete', 'Confirm Delete'])
        if 'search' in feature_lower:
            actions.extend(['Search', 'Filter', 'Sort'])
        if 'export' in feature_lower:
            actions.extend(['Export', 'Download'])

        return actions

class StatisticalAnalysisTools:
    """Statistical analysis tools for validation"""

    @staticmethod
    def perform_ab_test_analysis(control_group: List[float], test_group: List[float]) -> Dict[str, Any]:
        """Perform A/B test statistical analysis"""
        if len(control_group) < 2 or len(test_group) < 2:
            return {'error': 'Insufficient sample size for A/B testing'}

        # Calculate basic statistics
        control_mean = statistics.mean(control_group)
        test_mean = statistics.mean(test_group)
        control_std = statistics.stdev(control_group) if len(control_group) > 1 else 0
        test_std = statistics.stdev(test_group) if len(test_group) > 1 else 0

        # Calculate statistical significance (simplified t-test)
        pooled_std = ((control_std ** 2 / len(control_group)) + (test_std ** 2 / len(test_group))) ** 0.5
        t_statistic = (test_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        # Calculate p-value approximation (simplified)
        p_value = StatisticalAnalysisTools._approximate_p_value(abs(t_statistic))

        # Calculate confidence interval
        mean_difference = test_mean - control_mean
        standard_error = pooled_std
        confidence_interval = (
            mean_difference - 1.96 * standard_error,
            mean_difference + 1.96 * standard_error
        )

        return {
            'control_group': {
                'mean': round(control_mean, 3),
                'std_dev': round(control_std, 3),
                'sample_size': len(control_group)
            },
            'test_group': {
                'mean': round(test_mean, 3),
                'std_dev': round(test_std, 3),
                'sample_size': len(test_group)
            },
            'statistical_test': {
                't_statistic': round(t_statistic, 3),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05,
                'confidence_interval': (round(confidence_interval[0], 3), round(confidence_interval[1], 3))
            },
            'effect_size': {
                'mean_difference': round(mean_difference, 3),
                'relative_improvement': round((mean_difference / control_mean) * 100, 2) if control_mean != 0 else 0
            }
        }

    @staticmethod
    def _approximate_p_value(t_statistic: float) -> float:
        """Approximate p-value for t-statistic (simplified)"""
        # Using normal distribution approximation
        if t_statistic < 1.96:
            return 0.05
        elif t_statistic < 2.58:
            return 0.01
        elif t_statistic < 3.29:
            return 0.001
        else:
            return 0.0001

    @staticmethod
    def calculate_conversion_rates(events: List[Dict[str, Any]], funnel_stages: List[str]) -> Dict[str, Any]:
        """Calculate conversion rates through a funnel"""
        if not events or not funnel_stages:
            return {'error': 'Insufficient data for conversion analysis'}

        conversion_data = {}

        for i, stage in enumerate(funnel_stages):
            stage_events = [e for e in events if e.get('stage') == stage]
            stage_count = len(stage_events)

            if i == 0:
                conversion_data[stage] = {
                    'count': stage_count,
                    'conversion_rate': 1.0,
                    'drop_off_rate': 0.0
                }
            else:
                previous_stage = funnel_stages[i-1]
                previous_count = conversion_data[previous_stage]['count']

                conversion_rate = stage_count / previous_count if previous_count > 0 else 0
                drop_off_rate = 1 - conversion_rate

                conversion_data[stage] = {
                    'count': stage_count,
                    'conversion_rate': round(conversion_rate, 3),
                    'drop_off_rate': round(drop_off_rate, 3)
                }

        # Calculate overall metrics
        first_stage_count = conversion_data[funnel_stages[0]]['count']
        last_stage_count = conversion_data[funnel_stages[-1]]['count']

        overall_conversion = last_stage_count / first_stage_count if first_stage_count > 0 else 0

        return {
            'funnel_stages': conversion_data,
            'overall_conversion_rate': round(overall_conversion, 3),
            'funnel_efficiency': StatisticalAnalysisTools._calculate_funnel_efficiency(conversion_data),
            'bottlenecks': StatisticalAnalysisTools._identify_conversion_bottlenecks(conversion_data)
        }

    @staticmethod
    def _calculate_funnel_efficiency(conversion_data: Dict[str, Any]) -> float:
        """Calculate overall funnel efficiency"""
        conversion_rates = [stage_data['conversion_rate'] for stage_data in conversion_data.values()]

        if not conversion_rates:
            return 0.0

        # Geometric mean of conversion rates
        product = 1.0
        for rate in conversion_rates:
            product *= rate

        geometric_mean = product ** (1 / len(conversion_rates))
        return round(geometric_mean, 3)

    @staticmethod
    def _identify_conversion_bottlenecks(conversion_data: Dict[str, Any]) -> List[str]:
        """Identify bottlenecks in conversion funnel"""
        bottlenecks = []

        for stage, data in conversion_data.items():
            conversion_rate = data['conversion_rate']
            if conversion_rate < 0.5:  # Less than 50% conversion
                bottlenecks.append(f"High drop-off in {stage} ({(1-conversion_rate)*100:.1f}% drop-off)")

        return bottlenecks or ["No significant bottlenecks identified"]

    @staticmethod
    def analyze_time_series_data(time_series: List[Tuple[datetime, float]], analysis_type: str = 'trend') -> Dict[str, Any]:
        """Analyze time series data for trends and patterns"""
        if len(time_series) < 2:
            return {'error': 'Insufficient data points for time series analysis'}

        # Extract values and timestamps
        timestamps, values = zip(*time_series)

        analysis_result = {
            'data_points': len(time_series),
            'time_range': {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            },
            'basic_statistics': {
                'mean': round(statistics.mean(values), 3),
                'median': round(statistics.median(values), 3),
                'std_dev': round(statistics.stdev(values), 3) if len(values) > 1 else 0,
                'min_value': min(values),
                'max_value': max(values)
            }
        }

        if analysis_type == 'trend':
            analysis_result['trend_analysis'] = StatisticalAnalysisTools._calculate_trend(values)
        elif analysis_type == 'seasonal':
            analysis_result['seasonal_analysis'] = StatisticalAnalysisTools._analyze_seasonality(values)

        return analysis_result

    @staticmethod
    def _calculate_trend(values: List[float]) -> Dict[str, Any]:
        """Calculate trend in time series data"""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}

        # Simple linear regression
        n = len(values)
        x = list(range(n))

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi**2 for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2) if (n * sum_x2 - sum_x**2) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n

        # Determine trend direction
        if slope > 0.01:
            trend_direction = 'increasing'
        elif slope < -0.01:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'

        return {
            'trend_direction': trend_direction,
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'trend_strength': abs(slope),
            'r_squared': StatisticalAnalysisTools._calculate_r_squared(values, [intercept + slope * xi for xi in x])
        }

    @staticmethod
    def _calculate_r_squared(actual: List[float], predicted: List[float]) -> float:
        """Calculate R-squared value"""
        if len(actual) != len(predicted):
            return 0.0

        actual_mean = statistics.mean(actual)
        ss_total = sum((y - actual_mean)**2 for y in actual)
        ss_residual = sum((y - pred)**2 for y, pred in zip(actual, predicted))

        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        return round(r_squared, 3)

    @staticmethod
    def _analyze_seasonality(values: List[float]) -> Dict[str, Any]:
        """Analyze seasonality in time series data"""
        if len(values) < 7:  # Need at least a week of data
            return {'seasonality': 'insufficient_data'}

        # Simple autocorrelation analysis
        autocorr_values = []
        max_lag = min(len(values) // 2, 7)  # Check up to 7 days

        for lag in range(1, max_lag + 1):
            corr = StatisticalAnalysisTools._calculate_autocorrelation(values, lag)
            autocorr_values.append((lag, corr))

        # Find strongest correlation
        if autocorr_values:
            best_lag, best_corr = max(autocorr_values, key=lambda x: abs(x[1]))
            seasonality_strength = abs(best_corr)
        else:
            best_lag, best_corr = 0, 0
            seasonality_strength = 0

        return {
            'seasonality_detected': seasonality_strength > 0.3,
            'strongest_lag': best_lag,
            'seasonality_strength': round(seasonality_strength, 3),
            'autocorrelation_values': autocorr_values
        }

    @staticmethod
    def _calculate_autocorrelation(values: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(values) <= lag:
            return 0.0

        n = len(values)
        mean = statistics.mean(values)

        numerator = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag))
        denominator = sum((values[i] - mean)**2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0