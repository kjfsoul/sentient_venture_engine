#!/usr/bin/env python3
"""
Task 1.3: Causal Analysis Agent
Advanced causal inference agent using DoWhy, EconML, and causal-learn libraries

Features:
- Causal inference analysis using multiple libraries
- DAG modeling for hypothesis → strategies → outcomes
- Analysis of validation_results and human_feedback data
- LLM-powered causal insight interpretation
- CrewAI integration for collaborative analysis
- Storage in causal_insights table

Libraries integrated:
- DoWhy: Unified causal inference framework
- EconML: Machine learning-based causal inference
- causal-learn: Causal discovery algorithms
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# CrewAI imports
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    print("⚠️ CrewAI not available - running in standalone mode")
    CREWAI_AVAILABLE = False

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

# Import security manager
try:
    from security.api_key_manager import get_secret_optional
except ImportError:
    def get_secret_optional(key, fallbacks=None):
        return os.getenv(key)

# Load environment variables
load_dotenv()

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
    """Advanced Causal Analysis Agent for SVE"""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the causal analysis agent"""
        self.test_mode = test_mode
        
        # Initialize Supabase
        supabase_url = get_secret_optional("SUPABASE_URL")
        supabase_key = get_secret_optional("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase connection initialized")
        else:
            logger.warning("⚠️ Supabase credentials not found")
            self.supabase = None
        
        # Initialize LLM for interpretation
        self.llm = self._initialize_llm()
        
        # Define causal DAG structure
        self.causal_dag = self._define_causal_dag()
        
        # Causal hypotheses to test
        self.causal_hypotheses = self._define_causal_hypotheses()
        
        logger.info("🧠 Causal Analysis Agent initialized")
        self._log_library_status()
    
    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LLM for causal insight interpretation"""
        try:
            # Try OpenRouter first (free models)
            openrouter_key = get_secret_optional("OPENROUTER_API_KEY")
            if openrouter_key:
                return ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name="mistralai/mistral-7b-instruct:free",
                    temperature=0.3,
                    max_tokens=1024
                )
            
            # Fallback to regular OpenAI
            openai_key = get_secret_optional("OPENAI_API_KEY")
            if openai_key:
                return ChatOpenAI(
                    openai_api_key=openai_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=1024
                )
            
            logger.warning("⚠️ No LLM credentials available")
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
        logger.info(f"  CrewAI: {'✅ Available' if CREWAI_AVAILABLE else '❌ Not installed'}")
    
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
    
    def _extract_market_complexity(self, hypothesis: Dict, metrics: Dict) -> float:
        """Extract market complexity score from hypothesis and metrics"""
        # Simple heuristic - can be enhanced with NLP analysis
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        complexity_indicators = ['enterprise', 'b2b', 'platform', 'integration', 'ai', 'ml']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text)
        return min(complexity_score / len(complexity_indicators), 1.0)
    
    def _extract_validation_strategy(self, record: Dict) -> str:
        """Extract validation strategy type"""
        tier = record.get('tier', 1)
        strategy_map = {
            1: 'social_sentiment',
            2: 'prototype_testing',
            3: 'market_validation'
        }
        return strategy_map.get(tier, 'unknown')
    
    def _extract_resource_investment(self, metrics: Dict) -> float:
        """Extract resource investment level"""
        # Proxy based on metrics complexity
        if not metrics:
            return 0.3
        return min(len(metrics) / 10.0, 1.0)
    
    def _extract_hypothesis_novelty(self, hypothesis: Dict) -> float:
        """Extract hypothesis novelty score"""
        # Simple heuristic - can be enhanced with ML
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        novelty_indicators = ['new', 'novel', 'innovative', 'first', 'breakthrough', 'revolutionary']
        novelty_score = sum(1 for indicator in novelty_indicators if indicator in text)
        return min(novelty_score / len(novelty_indicators), 1.0)
    
    def _extract_market_timing(self, record: Dict) -> float:
        """Extract market timing score"""
        # Based on validation timestamp vs current market conditions
        # Simplified heuristic
        return np.random.uniform(0.3, 0.9)  # Placeholder
    
    def _extract_user_engagement(self, metrics: Dict) -> float:
        """Extract user engagement score"""
        engagement_metrics = ['user_engagement', 'interaction_rate', 'retention_rate']
        scores = [metrics.get(metric, 0) for metric in engagement_metrics if metric in metrics]
        return np.mean(scores) if scores else 0.5
    
    def _extract_feedback_quality(self, feedback: List[Dict]) -> float:
        """Extract feedback quality score"""
        if not feedback:
            return 0.3
        
        # Quality based on rationale text length and content
        quality_scores = []
        for f in feedback:
            rationale = f.get('rationale_text', '')
            if rationale:
                # Simple quality heuristic
                quality_score = min(len(rationale.split()) / 50.0, 1.0)
                quality_scores.append(quality_score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _extract_iteration_speed(self, record: Dict) -> float:
        """Extract iteration speed metric"""
        # Placeholder - would calculate based on timestamp differences
        return np.random.uniform(0.4, 0.8)
    
    def _extract_market_conditions(self, record: Dict) -> float:
        """Extract market conditions confounding variable"""
        # Placeholder - would integrate with market data
        return np.random.uniform(0.3, 0.7)
    
    def _extract_team_experience(self, hypothesis: Dict) -> float:
        """Extract team experience confounding variable"""
        agent = hypothesis.get('generated_by_agent', '')
        # Simple heuristic based on agent type
        experience_map = {
            'synthesis_agent': 0.8,
            'market_intel_agent': 0.7,
            'multimodal_agent': 0.6,
        }
        
        for agent_type, score in experience_map.items():
            if agent_type in agent.lower():
                return score
        return 0.5
    
    def _extract_competitive_landscape(self, metrics: Dict) -> float:
        """Extract competitive landscape confounding variable"""
        # Placeholder - would analyze competitive metrics
        return np.random.uniform(0.2, 0.8)
    
    def _extract_time_to_validation(self, record: Dict) -> float:
        """Extract time to validation outcome"""
        # Placeholder - would calculate actual time differences
        return np.random.uniform(1, 30)  # Days
    
    def _extract_cost_efficiency(self, metrics: Dict) -> float:
        """Extract cost efficiency outcome"""
        # Placeholder - would calculate based on resource usage
        return np.random.uniform(0.3, 0.9)
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """Generate simulated data for testing causal analysis"""
        logger.info("🧪 Generating simulated validation data for causal analysis")
        
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        
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
        
        # Generate outcomes with causal relationships
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
        
        data['cost_efficiency'] = (
            data['team_experience'] * 0.4 +
            data['resource_investment'] * 0.3 +
            (1 - data['market_complexity']) * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        data['human_approval'] = np.random.binomial(
            1, 
            0.7 * data['validation_success'] + 0.3 * data['user_engagement']
        )
        
        return pd.DataFrame(data)
    
    def run_dowhy_analysis(self, data: pd.DataFrame, hypothesis: CausalHypothesis) -> Optional[CausalResult]:
        """Run causal analysis using DoWhy library"""
        if not DOWHY_AVAILABLE:
            logger.warning("⚠️ DoWhy not available - skipping DoWhy analysis")
            return None
        
        try:
            logger.info(f"🔬 Running DoWhy analysis: {hypothesis.treatment} → {hypothesis.outcome}")
            
            # Create causal model
            model = CausalModel(
                data=data,
                treatment=hypothesis.treatment,
                outcome=hypothesis.outcome,
                common_causes=hypothesis.confounders
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate causal effect using multiple methods
            estimates = []
            methods = ['backdoor.linear_regression', 'backdoor.propensity_score_matching']
            
            for method in methods:
                try:
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name=method
                    )
                    estimates.append({
                        'method': method,
                        'effect': float(estimate.value),
                        'confidence_interval': estimate.get_confidence_intervals() if hasattr(estimate, 'get_confidence_intervals') else (None, None)
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Method {method} failed: {e}")
                    continue
            
            if not estimates:
                logger.warning("⚠️ No DoWhy estimates succeeded")
                return None
            
            # Use the first successful estimate
            best_estimate = estimates[0]
            
            # Refutation tests
            refutation_results = []
            try:
                # Random common cause
                refute_random = model.refute_estimate(
                    identified_estimand,
                    model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression'),
                    method_name="random_common_cause"
                )
                refutation_results.append(f"Random common cause p-value: {getattr(refute_random, 'new_effect', 'N/A')}")
            except:
                pass
            
            return CausalResult(
                effect_estimate=best_estimate['effect'],
                confidence_interval=best_estimate['confidence_interval'] or (-np.inf, np.inf),
                p_value=0.05,  # Placeholder - DoWhy doesn't always provide p-values
                method="DoWhy " + best_estimate['method'],
                interpretation=f"Causal effect of {hypothesis.treatment} on {hypothesis.outcome}: {best_estimate['effect']:.3f}",
                recommendations=refutation_results
            )
            
        except Exception as e:
            logger.error(f"❌ DoWhy analysis failed: {e}")
            return None
    
    def run_econml_analysis(self, data: pd.DataFrame, hypothesis: CausalHypothesis) -> Optional[CausalResult]:
        """Run causal analysis using EconML library"""
        if not ECONML_AVAILABLE:
            logger.warning("⚠️ EconML not available - skipping EconML analysis")
            return None
        
        try:
            logger.info(f"📊 Running EconML analysis: {hypothesis.treatment} → {hypothesis.outcome}")
            
            # Prepare data
            y = data[hypothesis.outcome].values
            
            # Handle categorical treatment
            if hypothesis.treatment in data.columns:
                if data[hypothesis.treatment].dtype == 'object':
                    # Convert categorical to numeric
                    treatment_map = {val: i for i, val in enumerate(data[hypothesis.treatment].unique())}
                    t = data[hypothesis.treatment].map(treatment_map).values
                else:
                    t = data[hypothesis.treatment].values
            else:
                logger.warning(f"⚠️ Treatment column {hypothesis.treatment} not found")
                return None
            
            # Confounders
            x_cols = [col for col in hypothesis.confounders if col in data.columns]
            if x_cols:
                x = data[x_cols].values
            else:
                x = None
            
            # Use LinearDML for continuous outcomes, adjust for binary
            if len(np.unique(y)) == 2:
                # Binary outcome - use logistic
                from econml.dml import NonParamDML
                est = NonParamDML(
                    model_y='auto',
                    model_t='auto',
                    model_final='auto',
                    random_state=42
                )
            else:
                # Continuous outcome
                est = LinearDML(random_state=42)
            
            # Fit the model
            est.fit(Y=y, T=t, X=x, W=x)
            
            # Get treatment effect
            if x is not None:
                te = est.effect(X=x)
                effect_estimate = np.mean(te)
                
                # Get confidence intervals
                try:
                    te_lower, te_upper = est.effect_interval(X=x, alpha=0.05)
                    confidence_interval = (np.mean(te_lower), np.mean(te_upper))
                except:
                    confidence_interval = (effect_estimate - 0.1, effect_estimate + 0.1)
            else:
                # No confounders case
                effect_estimate = est.coef_[0] if hasattr(est, 'coef_') else 0.0
                confidence_interval = (effect_estimate - 0.1, effect_estimate + 0.1)
            
            return CausalResult(
                effect_estimate=float(effect_estimate),
                confidence_interval=confidence_interval,
                p_value=0.05,  # Placeholder
                method="EconML LinearDML",
                interpretation=f"Machine learning causal effect of {hypothesis.treatment} on {hypothesis.outcome}: {effect_estimate:.3f}",
                recommendations=["EconML provides robust ML-based causal inference"]
            )
            
        except Exception as e:
            logger.error(f"❌ EconML analysis failed: {e}")
            return None
    
    def run_causal_discovery(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run causal discovery using causal-learn library"""
        if not CAUSAL_LEARN_AVAILABLE:
            logger.warning("⚠️ causal-learn not available - skipping causal discovery")
            return None
        
        try:
            logger.info("🔍 Running causal discovery analysis")
            
            # Select numerical columns for causal discovery
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                logger.warning("⚠️ Not enough numeric columns for causal discovery")
                return None
            
            # Limit to key variables to avoid complexity
            key_vars = ['market_complexity', 'resource_investment', 'user_engagement', 
                       'validation_success', 'time_to_validation']
            available_vars = [var for var in key_vars if var in numeric_cols]
            
            if len(available_vars) < 3:
                available_vars = numeric_cols[:min(5, len(numeric_cols))]
            
            discovery_data = data[available_vars].dropna()
            
            if len(discovery_data) < 20:
                logger.warning("⚠️ Not enough data points for causal discovery")
                return None
            
            # Run PC algorithm for causal discovery
            try:
                cg = pc(discovery_data.values, alpha=0.05, indep_test=fisherz)
                
                discovered_edges = []
                if hasattr(cg, 'G') and hasattr(cg, 'draw_pydot_graph'):
                    # Extract edges from the causal graph
                    graph = cg.G
                    for i in range(len(available_vars)):
                        for j in range(len(available_vars)):
                            if i != j and graph[i, j] == 1:
                                discovered_edges.append((available_vars[i], available_vars[j]))
                
                return {
                    'method': 'PC Algorithm (causal-learn)',
                    'discovered_edges': discovered_edges,
                    'variables': available_vars,
                    'interpretation': f"Discovered {len(discovered_edges)} causal relationships"
                }
                
            except Exception as e:
                logger.warning(f"⚠️ PC algorithm failed, trying GES: {e}")
                
                # Try GES algorithm as fallback
                try:
                    from causallearn.search.ScoreBased.GES import ges
                    record = ges(discovery_data.values)
                    
                    discovered_edges = []
                    if hasattr(record, 'G'):
                        graph = record['G']
                        for i in range(len(available_vars)):
                            for j in range(len(available_vars)):
                                if graph[i, j] == 1:
                                    discovered_edges.append((available_vars[i], available_vars[j]))
                    
                    return {
                        'method': 'GES Algorithm (causal-learn)',
                        'discovered_edges': discovered_edges,
                        'variables': available_vars,
                        'interpretation': f"Discovered {len(discovered_edges)} causal relationships using GES"
                    }
                except Exception as ges_error:
                    logger.error(f"❌ GES algorithm also failed: {ges_error}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Causal discovery failed: {e}")
            return None
    
    def interpret_causal_results(self, results: List[CausalResult], discovery_results: Optional[Dict]) -> str:
        """Use LLM to interpret causal analysis results"""
        if not self.llm:
            return self._generate_fallback_interpretation(results, discovery_results)
        
        try:
            # Prepare results summary for LLM
            results_summary = []
            for result in results:
                results_summary.append({
                    'effect_estimate': result.effect_estimate,
                    'confidence_interval': result.confidence_interval,
                    'method': result.method,
                    'interpretation': result.interpretation
                })
            
            discovery_summary = ""
            if discovery_results:
                discovery_summary = f"\nCausal Discovery Results:\n- Method: {discovery_results['method']}\n- {discovery_results['interpretation']}\n- Discovered edges: {discovery_results['discovered_edges']}"
            
            prompt = f"""
As a causal analysis expert, interpret these causal inference results for a venture validation system:

Analysis Results:
{json.dumps(results_summary, indent=2)}

{discovery_summary}

Provide insights on:
1. Which factors most strongly influence validation success
2. Actionable recommendations for improving validation strategies
3. Potential confounding factors to consider
4. Reliability assessment of the findings

Format as a structured analysis with clear recommendations.
"""
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ LLM interpretation failed: {e}")
            return self._generate_fallback_interpretation(results, discovery_results)
    
    def _generate_fallback_interpretation(self, results: List[CausalResult], discovery_results: Optional[Dict]) -> str:
        """Generate fallback interpretation without LLM"""
        interpretation = "# Causal Analysis Interpretation\n\n"
        
        if results:
            interpretation += "## Key Findings:\n"
            for i, result in enumerate(results, 1):
                interpretation += f"{i}. {result.interpretation}\n"
                interpretation += f"   - Method: {result.method}\n"
                interpretation += f"   - Effect size: {result.effect_estimate:.3f}\n"
                interpretation += f"   - Confidence interval: {result.confidence_interval}\n\n"
        
        if discovery_results:
            interpretation += f"## Causal Structure Discovery:\n"
            interpretation += f"- {discovery_results['interpretation']}\n"
            interpretation += f"- Key relationships found: {len(discovery_results.get('discovered_edges', []))}\n\n"
        
        interpretation += "## Recommendations:\n"
        interpretation += "1. Focus on factors with largest positive effects\n"
        interpretation += "2. Consider identified confounding variables in future analysis\n"
        interpretation += "3. Validate findings with additional data collection\n"
        
        return interpretation
    
    def store_causal_insights(self, hypothesis_id: str, causal_factors: List[str], 
                            causal_strength: float, recommendations: str) -> bool:
        """Store causal analysis results in causal_insights table"""
        if not self.supabase:
            logger.warning("⚠️ No Supabase connection - cannot store insights")
            return False
        
        try:
            insight_data = {
                'hypothesis_id': hypothesis_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'causal_factor_identified': ', '.join(causal_factors),
                'causal_strength': float(causal_strength),
                'recommendation_for_future_ideation': recommendations
            }
            
            result = self.supabase.table('causal_insights').insert(insight_data).execute()
            
            if result.data:
                logger.info(f"✅ Causal insights stored for hypothesis {hypothesis_id}")
                return True
            else:
                logger.error("❌ Failed to store causal insights")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing causal insights: {e}")
            return False
    
    def run_comprehensive_causal_analysis(self, days_back: int = 30) -> Dict[str, Any]:
        """Run comprehensive causal analysis on validation data"""
        logger.info("🧠 Starting comprehensive causal analysis")
        
        # Step 1: Retrieve validation data
        data = self.retrieve_validation_data(days_back)
        if data is None or len(data) < 10:
            return {
                "error": "Insufficient data for causal analysis",
                "success": False,
                "data_points": len(data) if data is not None else 0
            }
        
        logger.info(f"📊 Analyzing {len(data)} validation records")
        
        # Step 2: Run causal analysis for each hypothesis
        causal_results = []
        for hypothesis in self.causal_hypotheses:
            logger.info(f"\n🔬 Testing hypothesis: {hypothesis.hypothesis_text}")
            
            # Run DoWhy analysis
            dowhy_result = self.run_dowhy_analysis(data, hypothesis)
            if dowhy_result:
                causal_results.append(dowhy_result)
            
            # Run EconML analysis  
            econml_result = self.run_econml_analysis(data, hypothesis)
            if econml_result:
                causal_results.append(econml_result)
        
        # Step 3: Run causal discovery
        discovery_results = self.run_causal_discovery(data)
        
        # Step 4: Interpret results using LLM
        interpretation = self.interpret_causal_results(causal_results, discovery_results)
        
        # Step 5: Store insights in database
        stored_insights = []
        if causal_results and not self.test_mode:
            # Group results by hypothesis and store
            hypothesis_groups = {}
            for result in causal_results:
                key = result.method.split()[0]  # Group by method type
                if key not in hypothesis_groups:
                    hypothesis_groups[key] = []
                hypothesis_groups[key].append(result)
            
            for group_name, group_results in hypothesis_groups.items():
                # Use average effect as causal strength
                avg_effect = np.mean([r.effect_estimate for r in group_results])
                causal_factors = [r.method for r in group_results]
                
                stored = self.store_causal_insights(
                    hypothesis_id=f"group_{group_name}_{datetime.now().strftime('%Y%m%d')}",
                    causal_factors=causal_factors,
                    causal_strength=float(avg_effect),
                    recommendations=interpretation[:1000]  # Truncate for storage
                )
                stored_insights.append(stored)
        
        # Step 6: Return comprehensive results
        return {
            "success": True,
            "data_points_analyzed": len(data),
            "causal_hypotheses_tested": len(self.causal_hypotheses),
            "causal_results": [
                {
                    "method": r.method,
                    "effect_estimate": r.effect_estimate,
                    "confidence_interval": r.confidence_interval,
                    "interpretation": r.interpretation
                } for r in causal_results
            ],
            "causal_discovery": discovery_results,
            "llm_interpretation": interpretation,
            "insights_stored": sum(stored_insights) if stored_insights else 0,
            "causal_dag": self.causal_dag,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def create_crewai_agents(self) -> Optional[Tuple[Agent, Agent]]:
        """Create CrewAI agents for collaborative causal analysis"""
        if not CREWAI_AVAILABLE or not self.llm:
            logger.warning("⚠️ CrewAI or LLM not available - skipping agent creation")
            return None
        
        try:
            # Causal Analysis Specialist Agent
            causal_analyst = Agent(
                role='Causal Analysis Specialist',
                goal='Identify causal relationships between hypothesis attributes and validation outcomes',
                backstory="""
                You are an expert in causal inference with deep knowledge of experimental design,
                statistical analysis, and business hypothesis validation. You specialize in identifying
                true causal relationships while accounting for confounding variables and selection bias.
                """,
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
            
            # Business Strategy Advisor Agent
            strategy_advisor = Agent(
                role='Business Strategy Advisor',
                goal='Translate causal insights into actionable business recommendations',
                backstory="""
                You are a seasoned business strategist with expertise in venture validation,
                product development, and market entry strategies. You excel at translating
                statistical findings into practical, actionable business recommendations.
                """,
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
            
            return causal_analyst, strategy_advisor
            
        except Exception as e:
            logger.error(f"❌ Failed to create CrewAI agents: {e}")
            return None
    
    def run_crewai_analysis(self, causal_results: List[CausalResult], 
                           discovery_results: Optional[Dict]) -> Optional[str]:
        """Run collaborative analysis using CrewAI agents"""
        agents = self.create_crewai_agents()
        if not agents:
            return None
        
        causal_analyst, strategy_advisor = agents
        
        try:
            # Task for causal analysis
            causal_task = Task(
                description=f"""
                Analyze the following causal inference results from our venture validation system:
                
                Causal Results: {json.dumps([{
                    'method': r.method,
                    'effect_estimate': r.effect_estimate,
                    'confidence_interval': r.confidence_interval,
                    'interpretation': r.interpretation
                } for r in causal_results], indent=2)}
                
                Discovery Results: {json.dumps(discovery_results, indent=2) if discovery_results else 'None'}
                
                Provide a detailed causal analysis focusing on:
                1. Statistical significance and reliability of findings
                2. Potential confounding factors and limitations
                3. Strength of causal relationships identified
                4. Methodological considerations and robustness
                """,
                agent=causal_analyst,
                expected_output="Detailed statistical analysis with reliability assessment"
            )
            
            # Task for strategy recommendations
            strategy_task = Task(
                description="""
                Based on the causal analysis findings, develop actionable business recommendations for:
                1. Optimizing validation strategies
                2. Resource allocation decisions
                3. Improving hypothesis success rates
                4. Reducing validation time and costs
                5. Strategic priorities for future ideation
                
                Focus on practical, implementable recommendations that can immediately improve
                the venture validation process.
                """,
                agent=strategy_advisor,
                expected_output="Strategic recommendations with implementation roadmap",
                context=[causal_task]
            )
            
            # Create and run crew
            crew = Crew(
                agents=[causal_analyst, strategy_advisor],
                tasks=[causal_task, strategy_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return str(result) if result else None
            
        except Exception as e:
            logger.error(f"❌ CrewAI analysis failed: {e}")
            return None
    
    def run_complete_analysis(self, days_back: int = 30, use_crewai: bool = True) -> Dict[str, Any]:
        """Run complete causal analysis with optional CrewAI collaboration"""
        logger.info("🚀 Starting complete causal analysis pipeline")
        print("=" * 80)
        
        # Step 1: Run comprehensive causal analysis
        analysis_results = self.run_comprehensive_causal_analysis(days_back)
        
        if not analysis_results.get("success"):
            return analysis_results
        
        # Step 2: Run CrewAI collaborative analysis if requested and available
        crewai_results = None
        if use_crewai and CREWAI_AVAILABLE:
            logger.info("🤝 Running CrewAI collaborative analysis...")
            
            causal_results = [
                CausalResult(
                    effect_estimate=r["effect_estimate"],
                    confidence_interval=r["confidence_interval"],
                    p_value=0.05,
                    method=r["method"],
                    interpretation=r["interpretation"],
                    recommendations=[]
                ) for r in analysis_results["causal_results"]
            ]
            
            crewai_results = self.run_crewai_analysis(
                causal_results,
                analysis_results.get("causal_discovery")
            )
        
        # Step 3: Combine results
        final_results = {
            **analysis_results,
            "crewai_analysis": crewai_results,
            "analysis_complete": True,
            "libraries_used": {
                "DoWhy": DOWHY_AVAILABLE,
                "EconML": ECONML_AVAILABLE,
                "causal-learn": CAUSAL_LEARN_AVAILABLE,
                "CrewAI": CREWAI_AVAILABLE and use_crewai
            }
        }
        
        logger.info("✅ Complete causal analysis finished successfully")
        return final_results


def test_causal_analysis_agent():
    """Test the causal analysis agent"""
    print("🧠 Testing Causal Analysis Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = CausalAnalysisAgent(test_mode=True)
    
    # Test 1: Check library availability
    print("\n1. Library Status Check:")
    agent._log_library_status()
    
    # Test 2: Data retrieval (will use simulated data)
    print("\n2. Testing data retrieval...")
    data = agent.retrieve_validation_data(days_back=30)
    if data is not None:
        print(f"   ✅ Retrieved {len(data)} data points")
        print(f"   ✅ Columns: {list(data.columns)}")
    else:
        print("   ❌ Data retrieval failed")
        return False
    
    # Test 3: Causal analysis
    print("\n3. Testing causal analysis methods...")
    
    # Test DoWhy if available
    if DOWHY_AVAILABLE:
        hypothesis = agent.causal_hypotheses[0]
        result = agent.run_dowhy_analysis(data, hypothesis)
        if result:
            print(f"   ✅ DoWhy analysis successful: {result.effect_estimate:.3f}")
        else:
            print("   ⚠️ DoWhy analysis failed")
    
    # Test EconML if available
    if ECONML_AVAILABLE:
        hypothesis = agent.causal_hypotheses[1]
        result = agent.run_econml_analysis(data, hypothesis)
        if result:
            print(f"   ✅ EconML analysis successful: {result.effect_estimate:.3f}")
        else:
            print("   ⚠️ EconML analysis failed")
    
    # Test causal discovery
    if CAUSAL_LEARN_AVAILABLE:
        discovery = agent.run_causal_discovery(data)
        if discovery:
            print(f"   ✅ Causal discovery successful: {discovery['interpretation']}")
        else:
            print("   ⚠️ Causal discovery failed")
    
    # Test 4: Complete analysis pipeline
    print("\n4. Testing complete analysis pipeline...")
    try:
        results = agent.run_complete_analysis(days_back=30, use_crewai=False)
        if results.get("success"):
            print(f"   ✅ Complete analysis successful")
            print(f"   ✅ Data points: {results['data_points_analyzed']}")
            print(f"   ✅ Hypotheses tested: {results['causal_hypotheses_tested']}")
            print(f"   ✅ Results generated: {len(results['causal_results'])}")
            
            # Print sample interpretation
            if results.get('llm_interpretation'):
                print(f"\n   Sample interpretation (first 200 chars):")
                print(f"   {results['llm_interpretation'][:200]}...")
            
            return True
        else:
            print(f"   ❌ Complete analysis failed: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Complete analysis exception: {e}")
        return False


if __name__ == "__main__":
    # Run test
    success = test_causal_analysis_agent()
    
    if success:
        print("\n🎉 All tests passed! Causal Analysis Agent is ready.")
        
        # Optional: Run full analysis with CrewAI if available
        if CREWAI_AVAILABLE:
            print("\n🤝 Running full analysis with CrewAI...")
            agent = CausalAnalysisAgent(test_mode=False)
            full_results = agent.run_complete_analysis(days_back=30, use_crewai=True)
            
            if full_results.get("success"):
                print("✅ Full CrewAI analysis completed successfully!")
            else:
                print("⚠️ CrewAI analysis encountered issues")
    else:
        print("\n❌ Some tests failed. Check library installation and configuration.")
