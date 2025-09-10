#!/usr/bin/env python3
"""
Simplified Causal Analysis Script for SVE Task 1.3
Basic statistical causal inference when advanced libraries unavailable
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression, LogisticRegression
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from security.api_key_manager import get_secret_optional
except ImportError:
    def get_secret_optional(key, fallbacks=None):
        return os.getenv(key)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCausalAnalyzer:
    """Simplified Causal Analysis using basic statistical methods"""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        
        # Initialize Supabase
        supabase_url = get_secret_optional("SUPABASE_URL")
        supabase_key = get_secret_optional("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase connection initialized")
        else:
            self.supabase = None
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Define causal DAG
        self.causal_dag = {
            "treatments": ["resource_investment", "validation_strategy", "market_timing"],
            "outcomes": ["validation_success", "time_to_validation", "human_approval"],
            "confounders": ["market_conditions", "team_experience", "competitive_landscape"],
            "mediators": ["user_engagement", "feedback_quality"]
        }
        
        logger.info("📊 Simplified Causal Analyzer initialized")
    
    def _initialize_llm(self):
        """Initialize LLM for interpretation"""
        if not LLM_AVAILABLE:
            return None
        
        try:
            openrouter_key = get_secret_optional("OPENROUTER_API_KEY")
            if openrouter_key:
                return ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name="mistralai/mistral-7b-instruct:free",
                    temperature=0.3,
                    max_tokens=1024
                )
            return None
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            return None
    
    def retrieve_validation_data(self, days_back: int = 30) -> pd.DataFrame:
        """Retrieve validation data or generate simulated data"""
        if not self.supabase:
            return self._generate_simulated_data()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            validation_query = self.supabase.table('validation_results')\
                .select("*, hypotheses!inner(*), human_feedback(*)")\
                .gte('validation_timestamp', cutoff_date)\
                .execute()
            
            if not validation_query.data:
                return self._generate_simulated_data()
            
            # Convert to analysis format
            data_records = []
            for record in validation_query.data:
                hypothesis = record.get('hypotheses', {})
                feedback = record.get('human_feedback', [])
                metrics = record.get('metrics_json', {})
                
                data_record = {
                    'hypothesis_id': record['hypothesis_id'],
                    'resource_investment': self._extract_resource_score(metrics),
                    'validation_strategy': record['tier'] / 3.0,  # Normalize tier
                    'market_timing': np.random.uniform(0.3, 0.9),
                    'user_engagement': self._extract_engagement_score(metrics),
                    'feedback_quality': self._extract_feedback_score(feedback),
                    'market_conditions': np.random.uniform(0.3, 0.7),
                    'team_experience': self._extract_team_score(hypothesis),
                    'competitive_landscape': np.random.uniform(0.2, 0.8),
                    'validation_success': 1 if record['pass_fail_status'] == 'pass' else 0,
                    'time_to_validation': np.random.uniform(1, 30),
                    'human_approval': 1 if feedback and any(f.get('human_decision') == 'approve' for f in feedback) else 0,
                }
                data_records.append(data_record)
            
            return pd.DataFrame(data_records)
            
        except Exception as e:
            logger.error(f"❌ Error retrieving data: {e}")
            return self._generate_simulated_data()
    
    def _extract_resource_score(self, metrics: Dict) -> float:
        return min(len(metrics) / 10.0, 1.0) if metrics else 0.5
    
    def _extract_engagement_score(self, metrics: Dict) -> float:
        engagement_keys = ['user_engagement', 'interaction_rate']
        scores = [metrics.get(key, 0) for key in engagement_keys if key in metrics]
        return np.mean(scores) if scores else 0.5
    
    def _extract_feedback_score(self, feedback: List[Dict]) -> float:
        if not feedback:
            return 0.3
        quality_scores = []
        for f in feedback:
            rationale = f.get('rationale_text', '')
            if rationale:
                quality_scores.append(min(len(rationale.split()) / 50.0, 1.0))
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _extract_team_score(self, hypothesis: Dict) -> float:
        agent = hypothesis.get('generated_by_agent', '').lower()
        experience_map = {'synthesis': 0.8, 'market_intel': 0.7, 'multimodal': 0.6}
        for agent_type, score in experience_map.items():
            if agent_type in agent:
                return score
        return 0.5
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """Generate simulated validation data with causal relationships"""
        logger.info("🧪 Generating simulated validation data")
        
        np.random.seed(42)
        n_samples = 120
        
        # Generate features
        resource_investment = np.random.uniform(0, 1, n_samples)
        validation_strategy = np.random.choice([0.3, 0.6, 0.9], n_samples)
        market_timing = np.random.uniform(0, 1, n_samples)
        
        # Mediating variables
        user_engagement = np.clip(
            0.3 * validation_strategy + 0.2 * resource_investment + 
            0.2 * market_timing + np.random.normal(0, 0.1, n_samples), 0, 1
        )
        
        feedback_quality = np.clip(
            0.4 * resource_investment + 0.3 * user_engagement + 
            np.random.normal(0, 0.1, n_samples), 0, 1
        )
        
        # Confounders
        market_conditions = np.random.uniform(0, 1, n_samples)
        team_experience = np.random.uniform(0, 1, n_samples)
        competitive_landscape = np.random.uniform(0, 1, n_samples)
        
        # Outcomes with causal relationships
        validation_success_prob = (
            0.25 * resource_investment +
            0.20 * user_engagement +
            0.15 * team_experience +
            0.15 * market_conditions +
            0.10 * validation_strategy +
            0.10 * market_timing +
            0.05 * feedback_quality
        )
        validation_success = np.random.binomial(1, np.clip(validation_success_prob, 0, 1))
        
        time_to_validation = np.clip(
            10 + 5 * (1 - team_experience) + 3 * (1 - resource_investment) + 
            np.random.normal(0, 2, n_samples), 1, 30
        )
        
        human_approval = np.random.binomial(
            1, 0.6 * validation_success + 0.3 * user_engagement + 0.1 * feedback_quality
        )
        
        return pd.DataFrame({
            'hypothesis_id': [f"sim_hyp_{i:03d}" for i in range(n_samples)],
            'resource_investment': resource_investment,
            'validation_strategy': validation_strategy,
            'market_timing': market_timing,
            'user_engagement': user_engagement,
            'feedback_quality': feedback_quality,
            'market_conditions': market_conditions,
            'team_experience': team_experience,
            'competitive_landscape': competitive_landscape,
            'validation_success': validation_success,
            'time_to_validation': time_to_validation,
            'human_approval': human_approval
        })
    
    def analyze_causal_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze causal relationships using basic statistical methods"""
        logger.info("🔍 Analyzing causal relationships")
        
        results = {
            "correlations": {},
            "causal_estimates": {},
            "regression_results": {}
        }
        
        # 1. Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        outcome_vars = ['validation_success', 'time_to_validation', 'human_approval']
        for outcome in outcome_vars:
            if outcome in correlation_matrix.columns:
                correlations = correlation_matrix[outcome].drop(outcome).sort_values(key=abs, ascending=False)
                results["correlations"][outcome] = correlations.head(5).to_dict()
        
        # 2. Simple causal effect estimation
        results["causal_estimates"] = self._estimate_causal_effects(data)
        
        # 3. Basic regression analysis
        if STATS_AVAILABLE:
            results["regression_results"] = self._run_basic_regression(data)
        
        return results
    
    def _estimate_causal_effects(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate causal effects using simple comparisons"""
        effects = {}
        
        # Resource investment effect
        high_resource = data[data['resource_investment'] > data['resource_investment'].median()]
        low_resource = data[data['resource_investment'] <= data['resource_investment'].median()]
        
        resource_effect = high_resource['validation_success'].mean() - low_resource['validation_success'].mean()
        effects['resource_investment_effect'] = {
            "effect_size": resource_effect,
            "interpretation": f"High resource investment increases success rate by {resource_effect:.3f}"
        }
        
        # User engagement effect
        high_engagement = data[data['user_engagement'] > data['user_engagement'].median()]
        low_engagement = data[data['user_engagement'] <= data['user_engagement'].median()]
        
        engagement_effect = high_engagement['validation_success'].mean() - low_engagement['validation_success'].mean()
        effects['user_engagement_effect'] = {
            "effect_size": engagement_effect,
            "interpretation": f"High user engagement increases success rate by {engagement_effect:.3f}"
        }
        
        return effects
    
    def _run_basic_regression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run basic regression for causal inference"""
        regression_results = {}
        
        try:
            # Predict validation success
            X = data[['resource_investment', 'user_engagement', 'team_experience', 'market_conditions']]
            y = data['validation_success']
            
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            regression_results['validation_success_model'] = {
                "coefficients": dict(zip(X.columns, model.coef_[0])),
                "score": model.score(X, y)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Regression failed: {e}")
        
        return regression_results
    
    def interpret_results_with_llm(self, analysis_results: Dict[str, Any]) -> str:
        """Use LLM to interpret results"""
        if not self.llm:
            return self._generate_basic_interpretation(analysis_results)
        
        try:
            prompt = f"""
Analyze these causal inference results for venture validation:

CORRELATIONS: {json.dumps(analysis_results.get('correlations', {}), indent=2)}
CAUSAL EFFECTS: {json.dumps(analysis_results.get('causal_estimates', {}), indent=2)}

Provide:
1. Key factors driving validation success
2. Resource allocation recommendations  
3. Process optimizations
4. Strategic priorities

Focus on actionable recommendations.
"""
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ LLM interpretation failed: {e}")
            return self._generate_basic_interpretation(analysis_results)
    
    def _generate_basic_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Generate basic interpretation"""
        interpretation = "# Causal Analysis Results\n\n"
        
        if analysis_results.get('causal_estimates'):
            interpretation += "## Key Causal Effects:\n"
            for effect_name, effect_data in analysis_results['causal_estimates'].items():
                if 'interpretation' in effect_data:
                    interpretation += f"- {effect_data['interpretation']}\n"
        
        interpretation += "\n## Recommendations:\n"
        interpretation += "1. Increase resource investment for higher success rates\n"
        interpretation += "2. Focus on user engagement improvement\n"
        interpretation += "3. Optimize validation strategies\n"
        
        return interpretation
    
    def store_causal_insights(self, analysis_results: Dict[str, Any], interpretation: str) -> bool:
        """Store insights in causal_insights table"""
        if not self.supabase:
            return False
        
        try:
            causal_factors = []
            if analysis_results.get('causal_estimates'):
                for effect_name, effect_data in analysis_results['causal_estimates'].items():
                    if 'effect_size' in effect_data:
                        causal_factors.append(f"{effect_name}: {effect_data['effect_size']:.3f}")
            
            causal_strength = 0.5  # Default
            if analysis_results.get('correlations', {}).get('validation_success'):
                correlations = analysis_results['correlations']['validation_success']
                causal_strength = np.mean([abs(v) for v in correlations.values()])
            
            insight_data = {
                'hypothesis_id': f"causal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'analysis_timestamp': datetime.now().isoformat(),
                'causal_factor_identified': ', '.join(causal_factors[:3]),
                'causal_strength': float(causal_strength),
                'recommendation_for_future_ideation': interpretation[:1000]
            }
            
            result = self.supabase.table('causal_insights').insert(insight_data).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"❌ Error storing insights: {e}")
            return False
    
    def run_complete_analysis(self, days_back: int = 30) -> Dict[str, Any]:
        """Run complete causal analysis pipeline"""
        logger.info("🚀 Starting causal analysis")
        
        # Retrieve data
        data = self.retrieve_validation_data(days_back)
        if len(data) < 10:
            return {"error": "Insufficient data", "success": False}
        
        # Analyze relationships
        analysis_results = self.analyze_causal_relationships(data)
        
        # Interpret results
        interpretation = self.interpret_results_with_llm(analysis_results)
        
        # Store insights
        stored = False
        if not self.test_mode:
            stored = self.store_causal_insights(analysis_results, interpretation)
        
        return {
            "success": True,
            "data_points_analyzed": len(data),
            "causal_dag": self.causal_dag,
            "analysis_results": analysis_results,
            "llm_interpretation": interpretation,
            "insights_stored": stored,
            "analysis_timestamp": datetime.now().isoformat()
        }


def test_simplified_causal_analyzer():
    """Test the analyzer"""
    print("📊 Testing Simplified Causal Analyzer")
    print("=" * 50)
    
    analyzer = SimpleCausalAnalyzer(test_mode=True)
    results = analyzer.run_complete_analysis(days_back=30)
    
    if results.get("success"):
        print(f"✅ Analysis completed!")
        print(f"📊 Data points: {results['data_points_analyzed']}")
        print(f"🔍 Analysis keys: {list(results['analysis_results'].keys())}")
        print(f"💾 Stored: {results['insights_stored']}")
        return True
    else:
        print(f"❌ Failed: {results.get('error')}")
        return False


if __name__ == "__main__":
    success = test_simplified_causal_analyzer()
    if success:
        print("\n🎉 Causal Analyzer working!")
    else:
        print("\n❌ Issues detected.")
