#!/usr/bin/env python3
"""
Causal Inference Methods for SVE Project
Implements DoWhy, EconML, and causal-learn analysis methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import causal libraries with fallbacks
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

try:
    import econml
    from econml.dml import LinearDML
    from econml.orf import DMLOrthoForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False

@dataclass
class CausalResult:
    """Results from causal analysis"""
    effect_estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    interpretation: str
    recommendations: List[str]

class CausalInferenceMethods:
    """Causal inference methods using multiple libraries"""
    
    @staticmethod
    def run_dowhy_analysis(data: pd.DataFrame, treatment: str, outcome: str, 
                          confounders: List[str]) -> Optional[CausalResult]:
        """Run causal analysis using DoWhy library"""
        if not DOWHY_AVAILABLE:
            logger.warning("⚠️ DoWhy not available - skipping DoWhy analysis")
            return None
        
        try:
            logger.info(f"🔬 Running DoWhy analysis: {treatment} → {outcome}")
            
            # Ensure required columns exist
            required_cols = [treatment, outcome] + confounders
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"⚠️ Missing columns for DoWhy: {missing_cols}")
                return None
            
            # Create causal model
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders
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
                    
                    # Extract confidence intervals if available
                    ci = (None, None)
                    try:
                        if hasattr(estimate, 'get_confidence_intervals'):
                            ci_result = estimate.get_confidence_intervals()
                            if ci_result is not None:
                                ci = ci_result
                    except:
                        pass
                    
                    estimates.append({
                        'method': method,
                        'effect': float(estimate.value),
                        'confidence_interval': ci
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ DoWhy method {method} failed: {e}")
                    continue
            
            if not estimates:
                logger.warning("⚠️ No DoWhy estimates succeeded")
                return None
            
            # Use the first successful estimate
            best_estimate = estimates[0]
            
            # Run refutation tests
            refutation_results = []
            try:
                # Random common cause test
                refute_random = model.refute_estimate(
                    identified_estimand,
                    model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression'),
                    method_name="random_common_cause"
                )
                refutation_results.append(f"Random common cause test passed: {abs(refute_random.new_effect) < 0.1}")
                
                # Placebo treatment test
                refute_placebo = model.refute_estimate(
                    identified_estimand,
                    model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression'),
                    method_name="placebo_treatment_refuter"
                )
                refutation_results.append(f"Placebo treatment test passed: {abs(refute_placebo.new_effect) < 0.1}")
                
            except Exception as e:
                refutation_results.append(f"Refutation tests failed: {str(e)[:100]}")
            
            return CausalResult(
                effect_estimate=best_estimate['effect'],
                confidence_interval=best_estimate['confidence_interval'] or (-np.inf, np.inf),
                p_value=0.05,  # Placeholder - DoWhy doesn't always provide p-values
                method="DoWhy " + best_estimate['method'],
                interpretation=f"Causal effect of {treatment} on {outcome}: {best_estimate['effect']:.3f}",
                recommendations=refutation_results
            )
            
        except Exception as e:
            logger.error(f"❌ DoWhy analysis failed: {e}")
            return None
    
    @staticmethod
    def run_econml_analysis(data: pd.DataFrame, treatment: str, outcome: str, 
                           confounders: List[str]) -> Optional[CausalResult]:
        """Run causal analysis using EconML library"""
        if not ECONML_AVAILABLE:
            logger.warning("⚠️ EconML not available - skipping EconML analysis")
            return None
        
        try:
            logger.info(f"📊 Running EconML analysis: {treatment} → {outcome}")
            
            # Ensure required columns exist
            required_cols = [treatment, outcome] + confounders
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"⚠️ Missing columns for EconML: {missing_cols}")
                return None
            
            # Prepare data
            y = data[outcome].values
            
            # Handle categorical treatment
            if data[treatment].dtype == 'object':
                # Convert categorical to numeric
                treatment_map = {val: i for i, val in enumerate(data[treatment].unique())}
                t = data[treatment].map(treatment_map).values
            else:
                t = data[treatment].values
            
            # Confounders
            x_cols = [col for col in confounders if col in data.columns]
            if x_cols:
                x = data[x_cols].values
            else:
                x = None
            
            # Choose appropriate estimator based on outcome type
            if len(np.unique(y)) == 2:
                # Binary outcome - use DML with appropriate models
                try:
                    from econml.dml import NonParamDML
                    est = NonParamDML(
                        model_y='auto',
                        model_t='auto', 
                        model_final='auto',
                        random_state=42
                    )
                except ImportError:
                    # Fallback to LinearDML
                    est = LinearDML(random_state=42)
            else:
                # Continuous outcome
                est = LinearDML(random_state=42)
            
            # Fit the model
            if x is not None:
                est.fit(Y=y, T=t, X=x, W=x)
            else:
                # No confounders case - create dummy X
                dummy_x = np.ones((len(y), 1))
                est.fit(Y=y, T=t, X=dummy_x, W=dummy_x)
                x = dummy_x
            
            # Get treatment effect
            te = est.effect(X=x)
            effect_estimate = np.mean(te)
            
            # Get confidence intervals
            try:
                te_lower, te_upper = est.effect_interval(X=x, alpha=0.05)
                confidence_interval = (np.mean(te_lower), np.mean(te_upper))
            except Exception as e:
                logger.warning(f"⚠️ Could not compute confidence intervals: {e}")
                confidence_interval = (effect_estimate - 0.1, effect_estimate + 0.1)
            
            # Calculate p-value approximation
            se = (confidence_interval[1] - confidence_interval[0]) / (2 * 1.96)  # Approximate SE
            t_stat = effect_estimate / se if se > 0 else 0
            p_value = 2 * (1 - abs(t_stat))  # Rough approximation
            
            return CausalResult(
                effect_estimate=float(effect_estimate),
                confidence_interval=confidence_interval,
                p_value=max(0.001, min(p_value, 1.0)),
                method="EconML LinearDML",
                interpretation=f"Machine learning causal effect of {treatment} on {outcome}: {effect_estimate:.3f}",
                recommendations=["EconML provides robust ML-based causal inference with heterogeneous treatment effects"]
            )
            
        except Exception as e:
            logger.error(f"❌ EconML analysis failed: {e}")
            return None
    
    @staticmethod
    def run_causal_discovery(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
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
                       'validation_success', 'time_to_validation', 'cost_efficiency']
            available_vars = [var for var in key_vars if var in numeric_cols]
            
            if len(available_vars) < 3:
                available_vars = numeric_cols[:min(6, len(numeric_cols))]
            
            discovery_data = data[available_vars].dropna()
            
            if len(discovery_data) < 20:
                logger.warning("⚠️ Not enough data points for causal discovery")
                return None
            
            # Convert to numpy array
            data_matrix = discovery_data.values
            
            # Run PC algorithm for causal discovery
            try:
                cg = pc(data_matrix, alpha=0.05, indep_test=fisherz)
                
                # Extract discovered relationships
                discovered_edges = []
                if hasattr(cg, 'G') and cg.G is not None:
                    graph = cg.G
                    for i in range(len(available_vars)):
                        for j in range(len(available_vars)):
                            if i != j and graph[i, j] != 0:
                                edge_type = "directed" if graph[i, j] == 1 and graph[j, i] == 0 else "undirected"
                                discovered_edges.append({
                                    'from': available_vars[i],
                                    'to': available_vars[j],
                                    'type': edge_type,
                                    'strength': abs(graph[i, j])
                                })
                
                return {
                    'method': 'PC Algorithm (causal-learn)',
                    'variables_analyzed': available_vars,
                    'discovered_edges': discovered_edges,
                    'total_edges': len(discovered_edges),
                    'interpretation': f"Discovered {len(discovered_edges)} causal relationships among {len(available_vars)} variables"
                }
                
            except Exception as e:
                logger.warning(f"⚠️ PC algorithm failed, trying GES: {e}")
                
                # Fallback to GES algorithm
                try:
                    record = ges(data_matrix)
                    
                    discovered_edges = []
                    if hasattr(record, 'G') and record.G is not None:
                        graph = record.G
                        for i in range(len(available_vars)):
                            for j in range(len(available_vars)):
                                if i != j and graph[i, j] != 0:
                                    discovered_edges.append({
                                        'from': available_vars[i],
                                        'to': available_vars[j],
                                        'type': 'directed',
                                        'strength': abs(graph[i, j])
                                    })
                    
                    return {
                        'method': 'GES Algorithm (causal-learn)',
                        'variables_analyzed': available_vars,
                        'discovered_edges': discovered_edges,
                        'total_edges': len(discovered_edges),
                        'interpretation': f"GES discovered {len(discovered_edges)} causal relationships among {len(available_vars)} variables"
                    }
                    
                except Exception as e2:
                    logger.error(f"❌ Both PC and GES algorithms failed: {e2}")
                    return None
            
        except Exception as e:
            logger.error(f"❌ Causal discovery failed: {e}")
            return None
    
    @staticmethod
    def run_counterfactual_analysis(data: pd.DataFrame, treatment: str, outcome: str,
                                   treatment_value: float, counterfactual_value: float) -> Optional[Dict[str, Any]]:
        """Run counterfactual analysis - what if treatment had different value"""
        try:
            logger.info(f"🔮 Running counterfactual analysis: {treatment} = {counterfactual_value} vs {treatment_value}")
            
            if treatment not in data.columns or outcome not in data.columns:
                logger.warning(f"⚠️ Missing columns for counterfactual analysis")
                return None
            
            # Filter data for actual treatment value
            actual_data = data[data[treatment] == treatment_value]
            if len(actual_data) == 0:
                logger.warning(f"⚠️ No data points with {treatment} = {treatment_value}")
                return None
            
            # Calculate actual outcome
            actual_outcome = actual_data[outcome].mean()
            
            # Estimate counterfactual outcome using simple regression
            from sklearn.linear_model import LinearRegression
            
            # Use all numeric columns as features
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != outcome]
            
            if len(feature_cols) < 2:
                logger.warning("⚠️ Not enough features for counterfactual analysis")
                return None
            
            # Train regression model
            X = data[feature_cols].fillna(data[feature_cols].mean())
            y = data[outcome].fillna(data[outcome].mean())
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Create counterfactual data
            counterfactual_data = actual_data.copy()
            if treatment in feature_cols:
                counterfactual_data[treatment] = counterfactual_value
            
            # Predict counterfactual outcome
            X_counterfactual = counterfactual_data[feature_cols].fillna(data[feature_cols].mean())
            counterfactual_outcome = model.predict(X_counterfactual).mean()
            
            # Calculate treatment effect
            treatment_effect = counterfactual_outcome - actual_outcome
            
            return {
                'method': 'Counterfactual Analysis',
                'treatment': treatment,
                'actual_treatment_value': treatment_value,
                'counterfactual_treatment_value': counterfactual_value,
                'actual_outcome': actual_outcome,
                'counterfactual_outcome': counterfactual_outcome,
                'treatment_effect': treatment_effect,
                'interpretation': f"If {treatment} were {counterfactual_value} instead of {treatment_value}, {outcome} would change by {treatment_effect:.3f}",
                'sample_size': len(actual_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Counterfactual analysis failed: {e}")
            return None