# sentient_venture_engine/agents/analysis_agents.py
# Enhanced Analysis Agents for SVE Project including Causal Analysis

import os
import sys
import json
import requests
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from security.api_key_manager import get_secret_optional
from supabase import create_client

# Optional LangChain import with fallback
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain not available - LLM interpretation will be limited")
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None

# Causal inference libraries
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    print("⚠️ DoWhy not available - install with: pip install dowhy")
    DOWHY_AVAILABLE = False

try:
    import econml
    from econml.dml import LinearDML
    from econml.orf import DMLOrthoForest
    ECONML_AVAILABLE = True
except ImportError:
    print("⚠️ EconML not available - install with: pip install econml")
    ECONML_AVAILABLE = False

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    print("⚠️ causal-learn not available - install with: pip install causal-learn")
    CAUSAL_LEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CausalHypothesis:
    """Represents a causal hypothesis to be tested"""
    treatment: str
    outcome: str
    confounders: List[str]
    hypothesis_text: str
    expected_effect: str

@dataclass
class CausalResult:
    """Results from causal analysis"""
    effect_estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    interpretation: str
    recommendations: List[str]

class CausalAnalysisAgent:
    """Advanced Causal Analysis Agent for SVE - Identifies causal factors for hypothesis success/failure"""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the causal analysis agent"""
        self.test_mode = test_mode
        
        # Initialize Supabase (skip in test mode to avoid connection issues)
        if not test_mode:
            supabase_url = get_secret_optional('SUPABASE_URL')
            supabase_key = get_secret_optional('SUPABASE_KEY')
            
            if supabase_url and supabase_key:
                try:
                    self.supabase = create_client(supabase_url, supabase_key)
                    logger.info("✅ Supabase connection initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Supabase connection failed: {e}")
                    self.supabase = None
            else:
                logger.warning("⚠️ Supabase credentials not found in environment")
                self.supabase = None
        else:
            logger.info("🧪 Test mode: Skipping Supabase connection")
            self.supabase = None
        
        # Initialize LLM for interpretation (prioritizing cost-effective models)
        self.llm = self._initialize_llm()
        
        # Define causal DAG structure
        self.causal_dag = self._define_causal_dag()
        
        # Causal hypotheses to test
        self.causal_hypotheses = self._define_causal_hypotheses()
        
        logger.info("🧠 Causal Analysis Agent initialized")
        self._log_library_status()
    
    def _initialize_llm(self):
        """Initialize cost-effective LLM for causal insight interpretation"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️ LangChain not available - LLM interpretation disabled")
            return None
            
        try:
            # Prioritize OpenRouter free models for cost-effectiveness
            openrouter_key = get_secret_optional("OPENROUTER_API_KEY")
            if openrouter_key and ChatOpenAI:
                return ChatOpenAI(
                    model="qwen/qwen-2.5-7b-instruct:free",  # Qwen 3 equivalent
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    temperature=0.3,
                    max_tokens=2048,
                    default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE-CausalAnalysis"}
                )
            
            # Fallback to other cost-effective options
            gemini_key = get_secret_optional("GEMINI_API_KEY")
            if gemini_key and ChatOpenAI:
                return ChatOpenAI(
                    model="gemini-1.5-flash",
                    base_url="https://generativelanguage.googleapis.com/v1beta/",
                    api_key=gemini_key,
                    temperature=0.3,
                    max_tokens=2048
                )
            
            logger.warning("⚠️ No cost-effective LLM credentials available")
            return None
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            return None
    
    def _log_library_status(self):
        """Log the status of causal inference libraries"""
        logger.info("📚 Causal Inference Library Status:")
        logger.info(f"  DoWhy: {'✅ Available' if DOWHY_AVAILABLE else '❌ Not installed'}")
        logger.info(f"  EconML: {'✅ Available' if ECONML_AVAILABLE else '❌ Not installed'}")
        logger.info(f"  causal-learn: {'✅ Available' if CAUSAL_LEARN_AVAILABLE else '❌ Not installed'}")
    
    def _define_causal_dag(self) -> Dict[str, Any]:
        """Define the causal DAG linking hypothesis attributes → strategies → outcomes"""
        return {
            "nodes": {
                # Hypothesis attributes (treatments)
                "market_complexity": {"type": "treatment", "description": "Complexity of target market"},
                "validation_strategy": {"type": "treatment", "description": "Type of validation approach"},
                "resource_investment": {"type": "treatment", "description": "Resources allocated to validation"},
                "hypothesis_novelty": {"type": "treatment", "description": "How novel/innovative the hypothesis is"},
                "market_timing": {"type": "treatment", "description": "Market readiness timing"},
                
                # Intermediate variables (mediators)
                "user_engagement": {"type": "mediator", "description": "Level of user engagement achieved"},
                "feedback_quality": {"type": "mediator", "description": "Quality of feedback received"},
                "iteration_speed": {"type": "mediator", "description": "Speed of iteration cycles"},
                
                # Confounders
                "market_conditions": {"type": "confounder", "description": "External market conditions"},
                "team_experience": {"type": "confounder", "description": "Team experience level"},
                "competitive_landscape": {"type": "confounder", "description": "Competitive environment"},
                
                # Outcomes
                "validation_success": {"type": "outcome", "description": "Whether validation succeeded"},
                "time_to_validation": {"type": "outcome", "description": "Time taken to reach validation"},
                "cost_efficiency": {"type": "outcome", "description": "Cost efficiency of validation process"},
                "human_approval": {"type": "outcome", "description": "Human decision on hypothesis"}
            },
            
            "edges": [
                # Direct effects
                ("market_complexity", "validation_success"),
                ("validation_strategy", "validation_success"),
                ("resource_investment", "validation_success"),
                ("hypothesis_novelty", "validation_success"),
                ("market_timing", "validation_success"),
                
                # Mediated effects
                ("validation_strategy", "user_engagement"),
                ("user_engagement", "validation_success"),
                ("resource_investment", "feedback_quality"),
                ("feedback_quality", "validation_success"),
                ("market_timing", "iteration_speed"),
                ("iteration_speed", "time_to_validation"),
                
                # Confounding relationships
                ("market_conditions", "validation_success"),
                ("market_conditions", "market_timing"),
                ("team_experience", "validation_strategy"),
                ("team_experience", "validation_success"),
                ("competitive_landscape", "market_complexity"),
                ("competitive_landscape", "validation_success"),
                
                # Outcome relationships
                ("validation_success", "human_approval"),
                ("time_to_validation", "cost_efficiency"),
            ]
        }
    
    def _define_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Define causal hypotheses to test"""
        return [
            CausalHypothesis(
                treatment="validation_strategy",
                outcome="validation_success", 
                confounders=["market_conditions", "team_experience", "competitive_landscape"],
                hypothesis_text="Different validation strategies have different success rates",
                expected_effect="positive"
            ),
            CausalHypothesis(
                treatment="resource_investment",
                outcome="validation_success",
                confounders=["market_conditions", "team_experience"],
                hypothesis_text="Higher resource investment increases validation success probability",
                expected_effect="positive"
            ),
            CausalHypothesis(
                treatment="hypothesis_novelty",
                outcome="validation_success",
                confounders=["market_conditions", "competitive_landscape"],
                hypothesis_text="Novel hypotheses have different success patterns than incremental ones",
                expected_effect="uncertain"
            ),
            CausalHypothesis(
                treatment="market_timing",
                outcome="time_to_validation",
                confounders=["market_conditions", "competitive_landscape"],
                hypothesis_text="Market timing affects speed of validation",
                expected_effect="negative"
            ),
            CausalHypothesis(
                treatment="user_engagement",
                outcome="human_approval",
                confounders=["team_experience", "market_conditions"],
                hypothesis_text="User engagement level influences human approval decisions",
                expected_effect="positive"
            )
        ]
    
    def retrieve_validation_data(self, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Retrieve validation_results and human_feedback data from Supabase"""
        if not self.supabase:
            logger.warning("⚠️ No Supabase connection - using simulated data")
            return self._generate_simulated_data()
        
        try:
            # Get cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Query validation results with related data
            validation_query = self.supabase.table('validation_results')\
                .select("""
                    *,
                    hypotheses!inner(*),
                    human_feedback(*)
                """)\
                .gte('validation_timestamp', cutoff_date)\
                .execute()
            
            if not validation_query.data:
                logger.warning(f"⚠️ No validation data found in last {days_back} days")
                return self._generate_simulated_data()
            
            # Convert to DataFrame
            data_records = []
            for record in validation_query.data:
                # Extract features for causal analysis
                hypothesis = record.get('hypotheses', {})
                feedback = record.get('human_feedback', [])
                
                # Parse metrics
                metrics = record.get('metrics_json', {})
                
                data_record = {
                    'hypothesis_id': record['hypothesis_id'],
                    'validation_tier': record['tier'],
                    'pass_fail_status': record['pass_fail_status'],
                    'human_override_flag': record.get('human_override_flag', False),
                    'validation_timestamp': record['validation_timestamp'],
                    
                    # Hypothesis attributes
                    'initial_hypothesis_text': hypothesis.get('initial_hypothesis_text', ''),
                    'generated_by_agent': hypothesis.get('generated_by_agent', ''),
                    'validation_tier_progress': hypothesis.get('validation_tier_progress', 0),
                    
                    # Extracted features for causal analysis
                    'market_complexity': self._extract_market_complexity(hypothesis, metrics),
                    'validation_strategy': self._extract_validation_strategy(record),
                    'resource_investment': self._extract_resource_investment(metrics),
                    'hypothesis_novelty': self._extract_hypothesis_novelty(hypothesis),
                    'market_timing': self._extract_market_timing(record),
                    'user_engagement': self._extract_user_engagement(metrics),
                    'feedback_quality': self._extract_feedback_quality(feedback),
                    'iteration_speed': self._extract_iteration_speed(record),
                    
                    # Confounders
                    'market_conditions': self._extract_market_conditions(record),
                    'team_experience': self._extract_team_experience(hypothesis),
                    'competitive_landscape': self._extract_competitive_landscape(metrics),
                    
                    # Outcomes
                    'validation_success': 1 if record['pass_fail_status'] == 'pass' else 0,
                    'time_to_validation': self._extract_time_to_validation(record),
                    'cost_efficiency': self._extract_cost_efficiency(metrics),
                    'human_approval': 1 if feedback and any(f.get('human_decision') == 'approve' for f in feedback) else 0,
                }
                
                data_records.append(data_record)
            
            df = pd.DataFrame(data_records)
            logger.info(f"✅ Retrieved {len(df)} validation records for causal analysis")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving validation data: {e}")
            return self._generate_simulated_data()
    
    # Import feature extraction methods
    def _extract_market_complexity(self, hypothesis: Dict, metrics: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_market_complexity(hypothesis, metrics)
    
    def _extract_validation_strategy(self, record: Dict) -> str:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_validation_strategy(record)
    
    def _extract_resource_investment(self, metrics: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_resource_investment(metrics)
    
    def _extract_hypothesis_novelty(self, hypothesis: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_hypothesis_novelty(hypothesis)
    
    def _extract_market_timing(self, record: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_market_timing(record)
    
    def _extract_user_engagement(self, metrics: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_user_engagement(metrics)
    
    def _extract_feedback_quality(self, feedback: List[Dict]) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_feedback_quality(feedback)
    
    def _extract_iteration_speed(self, record: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_iteration_speed(record)
    
    def _extract_market_conditions(self, record: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_market_conditions(record)
    
    def _extract_team_experience(self, hypothesis: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_team_experience(hypothesis)
    
    def _extract_competitive_landscape(self, metrics: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_competitive_landscape(metrics)
    
    def _extract_time_to_validation(self, record: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_time_to_validation(record)
    
    def _extract_cost_efficiency(self, metrics: Dict) -> float:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.extract_cost_efficiency(metrics)
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        from .causal_analysis_methods import CausalAnalysisMethods
        return CausalAnalysisMethods.generate_simulated_data()
    
    def run_causal_analysis(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive causal analysis using multiple methods"""
        logger.info("🧠 Starting comprehensive causal analysis")
        
        # Get data if not provided
        if data is None:
            data = self.retrieve_validation_data()
            if data is None:
                logger.error("❌ No data available for causal analysis")
                return {'error': 'No data available'}
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_points': len(data),
            'causal_hypotheses_tested': [],
            'causal_discovery': None,
            'counterfactual_analyses': [],
            'llm_interpretation': None,
            'recommendations': [],
            'stored_insights': []
        }
        
        # Import causal inference methods
        from .causal_inference_methods import CausalInferenceMethods
        
        # Test each causal hypothesis
        for hypothesis in self.causal_hypotheses:
            logger.info(f"🔬 Testing hypothesis: {hypothesis.hypothesis_text}")
            
            hypothesis_results = {
                'hypothesis': hypothesis.hypothesis_text,
                'treatment': hypothesis.treatment,
                'outcome': hypothesis.outcome,
                'confounders': hypothesis.confounders,
                'methods': []
            }
            
            # Run DoWhy analysis
            dowhy_result = CausalInferenceMethods.run_dowhy_analysis(
                data, hypothesis.treatment, hypothesis.outcome, hypothesis.confounders
            )
            if dowhy_result:
                hypothesis_results['methods'].append({
                    'method': dowhy_result.method,
                    'effect_estimate': dowhy_result.effect_estimate,
                    'confidence_interval': dowhy_result.confidence_interval,
                    'p_value': dowhy_result.p_value,
                    'interpretation': dowhy_result.interpretation
                })
            
            # Run EconML analysis
            econml_result = CausalInferenceMethods.run_econml_analysis(
                data, hypothesis.treatment, hypothesis.outcome, hypothesis.confounders
            )
            if econml_result:
                hypothesis_results['methods'].append({
                    'method': econml_result.method,
                    'effect_estimate': econml_result.effect_estimate,
                    'confidence_interval': econml_result.confidence_interval,
                    'p_value': econml_result.p_value,
                    'interpretation': econml_result.interpretation
                })
            
            results['causal_hypotheses_tested'].append(hypothesis_results)
        
        # Run causal discovery
        discovery_result = CausalInferenceMethods.run_causal_discovery(data)
        if discovery_result:
            results['causal_discovery'] = discovery_result
        
        # Run counterfactual analyses
        counterfactual_scenarios = [
            ('resource_investment', 0.3, 0.8),  # What if we invested more resources?
            ('market_complexity', 0.8, 0.3),   # What if market was less complex?
            ('team_experience', 0.5, 0.9)      # What if team was more experienced?
        ]
        
        for treatment, actual_val, counterfactual_val in counterfactual_scenarios:
            if treatment in data.columns:
                cf_result = CausalInferenceMethods.run_counterfactual_analysis(
                    data, treatment, 'validation_success', actual_val, counterfactual_val
                )
                if cf_result:
                    results['counterfactual_analyses'].append(cf_result)
        
        # Generate LLM interpretation
        if self.llm:
            interpretation = self._generate_llm_interpretation(results)
            results['llm_interpretation'] = interpretation
            results['recommendations'] = self._extract_recommendations(interpretation)
        
        # Store insights in Supabase
        if not self.test_mode:
            stored_insights = self._store_causal_insights(results)
            results['stored_insights'] = stored_insights
        
        logger.info("✅ Causal analysis completed")
        return results
    
    def _generate_llm_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Generate natural language interpretation using LLM"""
        if not self.llm:
            return "LLM not available for interpretation"
        
        try:
            # Prepare analysis summary for LLM
            summary = f"""
            Causal Analysis Results Summary:
            
            Data Points Analyzed: {analysis_results['data_points']}
            
            Causal Hypotheses Tested:
            """
            
            for hyp in analysis_results['causal_hypotheses_tested']:
                summary += f"\n- {hyp['hypothesis']}"
                for method in hyp['methods']:
                    summary += f"\n  * {method['method']}: Effect = {method['effect_estimate']:.3f}"
            
            if analysis_results['causal_discovery']:
                discovery = analysis_results['causal_discovery']
                summary += f"\n\nCausal Discovery: {discovery['interpretation']}"
            
            if analysis_results['counterfactual_analyses']:
                summary += "\n\nCounterfactual Analyses:"
                for cf in analysis_results['counterfactual_analyses']:
                    summary += f"\n- {cf['interpretation']}"
            
            # LLM prompt for interpretation
            prompt = f"""
            As a causal analysis expert for the Sentient Venture Engine (SVE) project, analyze these causal inference results and provide actionable insights.
            
            {summary}
            
            Please provide:
            1. Key causal factors that drive hypothesis validation success
            2. Strength of each causal relationship (strong/moderate/weak)
            3. Actionable recommendations for improving "Time to First Dollar" (TTFD)
            4. Specific strategies for the synthesis crew to generate better hypotheses
            5. Risk factors to avoid in future hypothesis validation
            
            Focus on practical, implementable insights that can reduce TTFD to under 7 days.
            """
            
            response = self.llm.invoke(prompt)
            interpretation = response.content if hasattr(response, 'content') else str(response)
            
            return interpretation
            
        except Exception as e:
            logger.error(f"❌ LLM interpretation failed: {e}")
            return f"LLM interpretation failed: {str(e)}"
    
    def _extract_recommendations(self, interpretation: str) -> List[str]:
        """Extract actionable recommendations from LLM interpretation"""
        recommendations = []
        
        # Simple extraction based on common patterns
        lines = interpretation.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower() or 'strategy' in line.lower():
                in_recommendations = True
            elif in_recommendations and line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                recommendations.append(line.lstrip('•-*123456789. '))
            elif in_recommendations and line and not line.startswith(('•', '-', '*')) and not any(c.isdigit() for c in line[:3]):
                in_recommendations = False
        
        # Fallback: extract sentences with action words
        if not recommendations:
            action_words = ['should', 'must', 'need to', 'recommend', 'suggest', 'focus on', 'improve', 'increase', 'reduce']
            sentences = interpretation.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in action_words):
                    recommendations.append(sentence.strip())
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _store_causal_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Store causal insights in Supabase causal_insights table"""
        if not self.supabase:
            logger.warning("⚠️ No Supabase connection - cannot store insights")
            return []
        
        stored_insights = []
        
        try:
            # Store insights for each tested hypothesis
            for hyp_result in analysis_results['causal_hypotheses_tested']:
                for method_result in hyp_result['methods']:
                    
                    # Create a representative hypothesis_id (in real scenario, this would come from actual data)
                    hypothesis_id = f"causal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    insight_data = {
                        'hypothesis_id': hypothesis_id,
                        'causal_factor_identified': f"{hyp_result['treatment']} → {hyp_result['outcome']}",
                        'causal_strength': abs(method_result['effect_estimate']),
                        'recommendation_for_future_ideation': method_result['interpretation']
                    }
                    
                    result = self.supabase.table('causal_insights').insert(insight_data).execute()
                    
                    if result.data:
                        stored_insights.append(f"✅ Stored: {insight_data['causal_factor_identified']}")
                        logger.info(f"✅ Stored causal insight: {insight_data['causal_factor_identified']}")
                    else:
                        logger.warning(f"⚠️ Failed to store insight: {insight_data['causal_factor_identified']}")
            
            # Store causal discovery results
            if analysis_results['causal_discovery']:
                discovery = analysis_results['causal_discovery']
                for edge in discovery['discovered_edges']:
                    hypothesis_id = f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    insight_data = {
                        'hypothesis_id': hypothesis_id,
                        'causal_factor_identified': f"{edge['from']} → {edge['to']} ({edge['type']})",
                        'causal_strength': edge['strength'],
                        'recommendation_for_future_ideation': f"Discovered causal relationship via {discovery['method']}"
                    }
                    
                    result = self.supabase.table('causal_insights').insert(insight_data).execute()
                    
                    if result.data:
                        stored_insights.append(f"✅ Stored discovery: {insight_data['causal_factor_identified']}")
            
            return stored_insights
            
        except Exception as e:
            logger.error(f"❌ Error storing causal insights: {e}")
            return [f"❌ Storage failed: {str(e)}"]
    
    def generate_synthesis_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for the synthesis crew based on causal analysis"""
        logger.info("🎯 Generating synthesis crew recommendations")
        
        # Run causal analysis
        analysis_results = self.run_causal_analysis()
        
        if 'error' in analysis_results:
            return analysis_results
        
        # Extract key insights for synthesis crew
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'key_success_factors': [],
            'avoid_factors': [],
            'optimal_strategies': [],
            'resource_allocation_guidance': [],
            'market_timing_insights': [],
            'hypothesis_generation_guidelines': []
        }
        
        # Analyze causal results to extract recommendations
        for hyp_result in analysis_results['causal_hypotheses_tested']:
            treatment = hyp_result['treatment']
            outcome = hyp_result['outcome']
            
            # Calculate average effect across methods
            effects = [m['effect_estimate'] for m in hyp_result['methods']]
            if effects:
                avg_effect = np.mean(effects)
                
                if avg_effect > 0.1:  # Strong positive effect
                    recommendations['key_success_factors'].append({
                        'factor': treatment,
                        'effect_on': outcome,
                        'strength': avg_effect,
                        'recommendation': f"Prioritize {treatment} as it strongly improves {outcome}"
                    })
                elif avg_effect < -0.1:  # Strong negative effect
                    recommendations['avoid_factors'].append({
                        'factor': treatment,
                        'effect_on': outcome,
                        'strength': abs(avg_effect),
                        'recommendation': f"Minimize {treatment} as it negatively impacts {outcome}"
                    })
        
        # Add specific guidance based on analysis
        if analysis_results['llm_interpretation']:
            interpretation = analysis_results['llm_interpretation']
            
            # Extract strategy recommendations
            if 'resource' in interpretation.lower():
                recommendations['resource_allocation_guidance'].append(
                    "Optimize resource allocation based on causal analysis findings"
                )
            
            if 'timing' in interpretation.lower():
                recommendations['market_timing_insights'].append(
                    "Consider market timing factors in hypothesis validation"
                )
        
        logger.info("✅ Synthesis recommendations generated")
        return recommendations

class GitHubCodeAnalysisAgent:
    """Agent for analyzing GitHub repositories to identify technology trends and market insights."""
    
    def __init__(self):
        self.github_token = get_secret('GITHUB_TOKEN')
        self.supabase = create_client(get_secret('SUPABASE_URL'), get_secret('SUPABASE_KEY'))
        self.llm = self._initialize_llm()
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def _initialize_llm(self):
        """Initialize LLM for code analysis."""
        api_key = get_secret("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model="microsoft/phi-3-mini-128k-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
            default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE-CodeAnalysis"}
        )
    
    def search_trending_repositories(self, query: str = "created:>2024-01-01", limit: int = 20) -> List[Dict]:
        """Search for trending repositories based on criteria."""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': f'{query} stars:>100',
            'sort': 'stars',
            'order': 'desc',
            'per_page': limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            print(f"Error searching repositories: {e}")
            return []
    
    def analyze_repository_technologies(self, repo_data: Dict) -> Dict:
        """Analyze a repository to extract technology insights."""
        try:
            # Get language statistics
            languages_url = repo_data.get('languages_url')
            languages_response = requests.get(languages_url, headers=self.headers)
            languages = languages_response.json() if languages_response.status_code == 200 else {}
            
            # Get repository topics/tags
            topics = repo_data.get('topics', [])
            
            # Analyze with LLM
            analysis_prompt = f"""
            Analyze this GitHub repository for market intelligence:
            
            Repository: {repo_data.get('name')}
            Description: {repo_data.get('description', 'No description')}
            Stars: {repo_data.get('stargazers_count', 0)}
            Languages: {list(languages.keys())}
            Topics: {topics}
            
            Extract market insights in JSON format:
            {{
                "technology_trend": "Brief trend name",
                "market_signal": "What this indicates about market direction",
                "adoption_level": "emerging|growing|mature",
                "business_opportunity": "Potential business opportunity"
            }}
            
            Return only valid JSON.
            """
            
            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            try:
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis = json.loads(analysis_text[json_start:json_end])
                else:
                    analysis = self._create_fallback_analysis(repo_data, languages, topics)
            except json.JSONDecodeError:
                analysis = self._create_fallback_analysis(repo_data, languages, topics)
            
            # Add metadata
            analysis.update({
                'repository_name': repo_data.get('name'),
                'repository_url': repo_data.get('html_url'),
                'stars': repo_data.get('stargazers_count', 0),
                'primary_language': max(languages.keys(), key=languages.get) if languages else 'Unknown',
                'languages': list(languages.keys()),
                'topics': topics,
                'created_at': repo_data.get('created_at'),
                'updated_at': repo_data.get('updated_at')
            })
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing repository {repo_data.get('name')}: {e}")
            return self._create_fallback_analysis(repo_data, {}, [])
    
    def _create_fallback_analysis(self, repo_data: Dict, languages: Dict, topics: List) -> Dict:
        """Create fallback analysis when LLM fails."""
        primary_lang = max(languages.keys(), key=languages.get) if languages else 'Unknown'
        
        # Simple heuristics for technology trends
        if 'ai' in str(topics).lower() or 'ml' in str(topics).lower():
            trend = "AI/ML Integration"
            signal = "Growing AI adoption in software development"
        elif 'web3' in str(topics).lower() or 'blockchain' in str(topics).lower():
            trend = "Web3 Development"
            signal = "Blockchain technology adoption"
        elif primary_lang in ['TypeScript', 'JavaScript']:
            trend = "Modern Web Development"
            signal = "JavaScript ecosystem evolution"
        elif primary_lang == 'Python':
            trend = "Python Ecosystem Growth"
            signal = "Python's expanding use cases"
        else:
            trend = f"{primary_lang} Development"
            signal = f"Activity in {primary_lang} ecosystem"
        
        return {
            'technology_trend': trend,
            'market_signal': signal,
            'adoption_level': 'growing',
            'business_opportunity': f"Tools and services for {trend.lower()}"
        }
    
    def store_analysis_results(self, analyses: List[Dict]):
        """Store analysis results in Supabase."""
        for analysis in analyses:
            try:
                # Store in code_analysis table
                db_payload = {
                    'type': 'code_trend',
                    'source_url': analysis.get('repository_url', ''),
                    'processed_insights_path': f"{analysis.get('technology_trend')} - {analysis.get('market_signal')}",
                    'metadata': json.dumps({
                        'repository': analysis.get('repository_name'),
                        'stars': analysis.get('stars'),
                        'languages': analysis.get('languages'),
                        'topics': analysis.get('topics'),
                        'adoption_level': analysis.get('adoption_level'),
                        'business_opportunity': analysis.get('business_opportunity')
                    })
                }
                
                result = self.supabase.table('data_sources').insert(db_payload).execute()
                print(f"✅ STORED CODE TREND: {analysis.get('technology_trend')}")
                
            except Exception as e:
                print(f"⚠️ Failed to store analysis for {analysis.get('repository_name')}: {e}")
    
    def run_code_analysis(self, search_queries: List[str] = None) -> Dict:
        """Run comprehensive code analysis for market intelligence."""
        if search_queries is None:
            search_queries = [
                "created:>2024-01-01 topic:ai",
                "created:>2024-01-01 topic:saas", 
                "created:>2024-01-01 language:python",
                "created:>2024-01-01 topic:startup"
            ]
        
        all_analyses = []
        
        for query in search_queries:
            print(f"🔍 Searching repositories: {query}")
            repos = self.search_trending_repositories(query, limit=5)
            
            for repo in repos:
                analysis = self.analyze_repository_technologies(repo)
                if analysis:
                    all_analyses.append(analysis)
        
        # Store results
        if all_analyses:
            self.store_analysis_results(all_analyses)
        
        print(f"📊 Code Analysis Complete: Processed {len(all_analyses)} repositories")
        return {
            'total_analyzed': len(all_analyses),
            'trends_identified': len(set(a.get('technology_trend') for a in all_analyses)),
            'analyses': all_analyses[:5]  # Return sample for verification
        }

def run_code_intelligence():
    """Main function to run code intelligence analysis."""
    print("🚀 Starting GitHub Code Analysis Agent...")
    
    try:
        agent = GitHubCodeAnalysisAgent()
        results = agent.run_code_analysis()
        
        print("\n--- Code Intelligence Results ---")
        print(f"Repositories analyzed: {results['total_analyzed']}")
        print(f"Unique trends identified: {results['trends_identified']}")
        print("\nSample trends:")
        for analysis in results['analyses']:
            print(f"  • {analysis.get('technology_trend')}: {analysis.get('market_signal')}")
        
        print("\n✅ Code Analysis Complete")
        return results
        
    except Exception as e:
        print(f"❌ Code analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    run_code_intelligence()
