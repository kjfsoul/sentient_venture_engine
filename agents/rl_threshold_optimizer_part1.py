#!/usr/bin/env python3
"""
Task 1.4: Dynamic Threshold Adjustment - Part 1
RL Environment and Basic Infrastructure

Part 1 includes:
- ValidationThresholdEnv (Gymnasium environment) 
- State-action-reward framework
- Data preprocessing utilities
- Basic execution safeguards

Libraries: Stable Baselines3, Gymnasium, NumPy, Pandas
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Import security manager
try:
    from security.api_key_manager import get_secret_optional
except ImportError:
    def get_secret_optional(key, fallbacks=None):
        return os.getenv(key)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThresholdState:
    """Current threshold state for validation tiers"""
    social_sentiment_threshold: float = 0.6
    prototype_testing_threshold: float = 0.7
    market_validation_threshold: float = 0.8
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.social_sentiment_threshold,
            self.prototype_testing_threshold,
            self.market_validation_threshold
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ThresholdState':
        return cls(
            social_sentiment_threshold=float(arr[0]),
            prototype_testing_threshold=float(arr[1]),
            market_validation_threshold=float(arr[2])
        )

@dataclass
class ValidationOutcome:
    """Outcome of validation with current thresholds"""
    tier_passed: int  # 0, 1, 2, 3
    validation_success: bool
    time_to_complete: float
    resource_cost: float
    first_dollar_achieved: bool
    human_approval: bool

class ValidationThresholdEnv(gym.Env):
    """
    Gymnasium environment for RL-based dynamic threshold adjustment
    
    State: [hypothesis_features(6) + market_conditions(3) + thresholds(3)] = 12D
    Action: Threshold adjustments [-0.2, +0.2] for 3 tiers = 3D continuous
    Reward: +1000 for first dollar + efficiency bonuses + validation success
    """
    
    def __init__(self, validation_data: pd.DataFrame, max_episodes: int = 1000):
        super().__init__()
        
        self.validation_data = validation_data
        self.max_episodes = max_episodes
        self.episode_count = 0
        
        # State space: 12D continuous
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Action space: 3D continuous threshold adjustments
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)
        
        # Initialize state
        self.current_thresholds = ThresholdState()
        self.current_hypothesis_idx = 0
        self.episode_steps = 0
        self.max_steps_per_episode = 50
        
        # Performance tracking
        self.performance_history = []
        self.first_dollar_count = 0
        
        logger.info(f"🏋️ ValidationThresholdEnv initialized with {len(validation_data)} data points")
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.episode_steps = 0
        
        # Execution safeguard
        if self.episode_count > self.max_episodes:
            self.episode_count = 0
        
        self.current_thresholds = ThresholdState()
        self.current_hypothesis_idx = np.random.randint(0, len(self.validation_data))
        
        state = self._get_current_state()
        info = {
            "episode": self.episode_count,
            "hypothesis_idx": self.current_hypothesis_idx,
            "thresholds": self.current_thresholds.to_array()
        }
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        self.episode_steps += 1
        
        # Apply threshold adjustments with bounds checking
        old_thresholds = self.current_thresholds.to_array().copy()
        new_thresholds = old_thresholds + action
        
        # Enforce bounds [0.1, 0.95]
        new_thresholds = np.clip(new_thresholds, 0.1, 0.95)
        
        # Ensure increasing order: tier1 <= tier2 <= tier3
        new_thresholds = self._enforce_threshold_ordering(new_thresholds)
        
        self.current_thresholds = ThresholdState.from_array(new_thresholds)
        
        # Simulate validation outcome
        outcome = self._simulate_validation_outcome()
        
        # Calculate reward
        reward = self._calculate_reward(outcome, action)
        
        # Check termination
        terminated = (
            outcome.validation_success or 
            outcome.tier_passed == 0 or
            self.episode_steps >= self.max_steps_per_episode
        )
        
        truncated = self.episode_steps >= self.max_steps_per_episode
        
        new_state = self._get_current_state()
        
        info = {
            "outcome": outcome,
            "old_thresholds": old_thresholds,
            "new_thresholds": new_thresholds,
            "episode_steps": self.episode_steps,
            "first_dollar": outcome.first_dollar_achieved
        }
        
        self._update_performance_tracking(outcome, reward)
        
        return new_state, reward, terminated, truncated, info
    
    def _get_current_state(self) -> np.ndarray:
        """Get current environment state (12D vector)"""
        if self.current_hypothesis_idx >= len(self.validation_data):
            self.current_hypothesis_idx = 0
        
        hypothesis_data = self.validation_data.iloc[self.current_hypothesis_idx]
        
        # Extract hypothesis features (6D)
        hypothesis_features = np.array([
            float(hypothesis_data.get('market_complexity', 0.5)),
            self._encode_validation_strategy(hypothesis_data.get('validation_strategy', 'social_sentiment')),
            float(hypothesis_data.get('resource_investment', 0.5)),
            float(hypothesis_data.get('hypothesis_novelty', 0.5)),
            float(hypothesis_data.get('market_timing', 0.5)),
            float(hypothesis_data.get('user_engagement', 0.5))
        ], dtype=np.float32)
        
        # Market conditions (3D)
        market_conditions = np.array([0.5, 0.5, 0.7], dtype=np.float32)
        
        # Current thresholds (3D)
        current_thresholds = self.current_thresholds.to_array()
        
        # Combine into 12D state
        state = np.concatenate([hypothesis_features, market_conditions, current_thresholds])
        return state.astype(np.float32)
    
    def _encode_validation_strategy(self, strategy: str) -> float:
        """Encode validation strategy as numeric"""
        strategy_map = {
            'social_sentiment': 0.2,
            'prototype_testing': 0.5,
            'market_validation': 0.8,
            'unknown': 0.5
        }
        return strategy_map.get(strategy, 0.5)
    
    def _enforce_threshold_ordering(self, thresholds: np.ndarray) -> np.ndarray:
        """Ensure tier1 <= tier2 <= tier3"""
        ordered = thresholds.copy()
        
        if ordered[1] < ordered[0]:
            ordered[1] = ordered[0] + 0.05
        if ordered[2] < ordered[1]:
            ordered[2] = ordered[1] + 0.05
        
        return np.clip(ordered, 0.1, 0.95)
    
    def _simulate_validation_outcome(self) -> ValidationOutcome:
        """Simulate validation outcome based on current thresholds"""
        hypothesis_data = self.validation_data.iloc[self.current_hypothesis_idx]
        
        base_success_prob = float(hypothesis_data.get('validation_success', 0.5))
        base_engagement = float(hypothesis_data.get('user_engagement', 0.5))
        
        thresholds = self.current_thresholds.to_array()
        
        # Simulate tier progression
        tier_passed = 0
        validation_success = False
        
        # Tier 1: Social Sentiment
        tier1_score = base_engagement + np.random.normal(0, 0.1)
        if tier1_score >= thresholds[0]:
            tier_passed = 1
            
            # Tier 2: Prototype Testing
            tier2_score = base_success_prob + np.random.normal(0, 0.1)
            if tier2_score >= thresholds[1]:
                tier_passed = 2
                
                # Tier 3: Market Validation
                tier3_score = (base_success_prob + base_engagement) / 2 + np.random.normal(0, 0.1)
                if tier3_score >= thresholds[2]:
                    tier_passed = 3
                    validation_success = True
        
        # Calculate outcomes
        time_to_complete = max(1.0, 5.0 + (3 - tier_passed) * 5.0 + np.random.exponential(2.0))
        
        threshold_avg = np.mean(thresholds)
        resource_cost = min(0.5 * (2.0 - threshold_avg) + np.random.uniform(0, 0.2), 1.0)
        
        first_dollar_achieved = validation_success and np.random.random() < 0.7
        
        human_approval_prob = 0.3 + 0.6 * validation_success + 0.1 * base_engagement
        human_approval = np.random.random() < human_approval_prob
        
        return ValidationOutcome(
            tier_passed=tier_passed,
            validation_success=validation_success,
            time_to_complete=time_to_complete,
            resource_cost=resource_cost,
            first_dollar_achieved=first_dollar_achieved,
            human_approval=human_approval
        )
    
    def _calculate_reward(self, outcome: ValidationOutcome, action: np.ndarray) -> float:
        """Calculate reward: +1000 for first dollar + efficiency bonuses"""
        reward = 0.0
        
        # Primary reward: First dollar achievement (+1000)
        if outcome.first_dollar_achieved:
            reward += 1000.0
            self.first_dollar_count += 1
        
        # Validation success bonus (+500)
        if outcome.validation_success:
            reward += 500.0
        elif outcome.tier_passed >= 2:
            reward += 200.0
        elif outcome.tier_passed >= 1:
            reward += 100.0
        
        # Efficiency bonuses
        if outcome.time_to_complete <= 7.0:
            reward += 100.0 * (1.0 - outcome.time_to_complete / 7.0)
        
        reward += 50.0 * (1.0 - outcome.resource_cost)
        
        if outcome.human_approval:
            reward += 150.0
        
        # Penalties for large threshold changes
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 0.15:
            reward -= 25.0 * action_magnitude
        
        return reward
    
    def _update_performance_tracking(self, outcome: ValidationOutcome, reward: float):
        """Update performance tracking"""
        self.performance_history.append({
            'episode': self.episode_count,
            'step': self.episode_steps,
            'tier_passed': outcome.tier_passed,
            'validation_success': outcome.validation_success,
            'first_dollar': outcome.first_dollar_achieved,
            'reward': reward,
            'thresholds': self.current_thresholds.to_array().copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep recent history only (execution safeguard)
        if len(self.performance_history) > 5000:
            self.performance_history = self.performance_history[-2500:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-100:] if len(self.performance_history) >= 100 else self.performance_history
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': len(self.performance_history),
            'first_dollar_count': self.first_dollar_count,
            'recent_success_rate': np.mean([h['validation_success'] for h in recent_history]),
            'recent_avg_reward': np.mean([h['reward'] for h in recent_history]),
            'recent_avg_tiers_passed': np.mean([h['tier_passed'] for h in recent_history]),
            'first_dollar_rate': self.first_dollar_count / max(1, self.episode_count),
            'current_thresholds': self.current_thresholds.to_array().tolist()
        }


class ValidationDataProcessor:
    """Utility class for processing Supabase validation data for RL training"""
    
    def __init__(self):
        supabase_url = get_secret_optional("SUPABASE_URL")
        supabase_key = get_secret_optional("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase connection initialized")
        else:
            logger.warning("⚠️ Supabase credentials not found - will use simulated data")
            self.supabase = None
    
    def retrieve_training_data(self, days_back: int = 90) -> pd.DataFrame:
        """Retrieve and process validation data for RL training"""
        if not self.supabase:
            return self._generate_simulated_training_data()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            validation_query = self.supabase.table('validation_results')\
                .select("*, hypotheses!inner(*), human_feedback(*)")\
                .gte('validation_timestamp', cutoff_date)\
                .execute()
            
            if not validation_query.data:
                return self._generate_simulated_training_data()
            
            processed_data = self._process_validation_records(validation_query.data)
            logger.info(f"✅ Retrieved {len(processed_data)} validation records for RL training")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving validation data: {e}")
            return self._generate_simulated_training_data()
    
    def _process_validation_records(self, records: List[Dict]) -> pd.DataFrame:
        """Process raw validation records into RL training format"""
        processed_records = []
        
        for record in records:
            hypothesis = record.get('hypotheses', {})
            feedback = record.get('human_feedback', [])
            metrics = record.get('metrics_json', {})
            
            processed_record = {
                'hypothesis_id': record['hypothesis_id'],
                'validation_tier': record['tier'],
                'market_complexity': self._extract_feature(hypothesis, 'market_complexity'),
                'validation_strategy': self._extract_validation_strategy(record),
                'resource_investment': min(len(metrics) / 10.0, 1.0) if metrics else 0.5,
                'hypothesis_novelty': self._extract_novelty(hypothesis),
                'market_timing': np.random.uniform(0.3, 0.9),  # Placeholder
                'user_engagement': self._extract_engagement(metrics),
                'validation_success': 1 if record['pass_fail_status'] == 'pass' else 0,
                'human_approval': 1 if feedback and any(f.get('human_decision') == 'approve' for f in feedback) else 0
            }
            
            processed_records.append(processed_record)
        
        return pd.DataFrame(processed_records)
    
    def _extract_feature(self, hypothesis: Dict, feature_name: str) -> float:
        """Extract feature with fallback"""
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        if feature_name == 'market_complexity':
            indicators = ['enterprise', 'b2b', 'platform', 'integration']
            score = sum(1 for indicator in indicators if indicator in text)
            return min(score / len(indicators), 1.0)
        return 0.5
    
    def _extract_validation_strategy(self, record: Dict) -> str:
        """Extract validation strategy"""
        tier = record.get('tier', 1)
        strategy_map = {1: 'social_sentiment', 2: 'prototype_testing', 3: 'market_validation'}
        return strategy_map.get(tier, 'social_sentiment')
    
    def _extract_novelty(self, hypothesis: Dict) -> float:
        """Extract hypothesis novelty"""
        text = hypothesis.get('initial_hypothesis_text', '').lower()
        novelty_indicators = ['new', 'novel', 'innovative', 'first']
        score = sum(1 for indicator in novelty_indicators if indicator in text)
        return min(score / len(novelty_indicators), 1.0)
    
    def _extract_engagement(self, metrics: Dict) -> float:
        """Extract user engagement"""
        engagement_keys = ['engagement', 'interaction', 'retention']
        scores = []
        
        for key, value in metrics.items():
            if any(eng_key in key.lower() for eng_key in engagement_keys):
                try:
                    scores.append(float(value))
                except (ValueError, TypeError):
                    continue
        
        return np.mean(scores) if scores else 0.5
    
    def _generate_simulated_training_data(self) -> pd.DataFrame:
        """Generate simulated training data"""
        logger.info("🧪 Generating simulated RL training data")
        
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'hypothesis_id': [f"hyp_rl_{i:03d}" for i in range(n_samples)],
            'validation_tier': np.random.choice([1, 2, 3], n_samples),
            'market_complexity': np.random.uniform(0, 1, n_samples),
            'validation_strategy': np.random.choice(['social_sentiment', 'prototype_testing', 'market_validation'], n_samples),
            'resource_investment': np.random.uniform(0, 1, n_samples),
            'hypothesis_novelty': np.random.uniform(0, 1, n_samples),
            'market_timing': np.random.uniform(0, 1, n_samples),
            'user_engagement': np.random.uniform(0, 1, n_samples)
        }
        
        # Generate outcomes with realistic relationships
        success_prob = (
            0.3 * np.array(data['resource_investment']) +
            0.2 * np.array(data['user_engagement']) +
            0.3 * np.array(data['market_timing']) +
            0.2 * (1 - np.array(data['market_complexity']))
        )
        
        data['validation_success'] = np.random.binomial(1, success_prob)
        data['human_approval'] = np.random.binomial(1, 0.7 * np.array(data['validation_success']) + 0.2)
        
        return pd.DataFrame(data)


def test_rl_environment_part1():
    """Test the ValidationThresholdEnv - Part 1"""
    print("🏋️ Testing ValidationThresholdEnv - Part 1")
    print("=" * 60)
    
    # Create training data
    print("\n1. Creating training data...")
    processor = ValidationDataProcessor()
    training_data = processor.retrieve_training_data(days_back=90)
    print(f"   ✅ Training data: {len(training_data)} samples")
    
    # Initialize environment
    print("\n2. Initializing RL environment...")
    env = ValidationThresholdEnv(training_data, max_episodes=10)
    print(f"   ✅ Environment initialized")
    print(f"   ✅ Observation space: {env.observation_space.shape}")
    print(f"   ✅ Action space: {env.action_space.shape}")
    
    # Test environment
    print("\n3. Testing environment...")
    state, info = env.reset()
    print(f"   ✅ Reset successful, state shape: {state.shape}")
    
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"   Step {step+1}: reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"   ✅ Total reward: {total_reward:.2f}")
    
    # Performance summary
    summary = env.get_performance_summary()
    print(f"\n4. Performance Summary:")
    print(f"   ✅ Episodes: {summary.get('total_episodes', 0)}")
    print(f"   ✅ First dollars: {summary.get('first_dollar_count', 0)}")
    
    return True


if __name__ == "__main__":
    success = test_rl_environment_part1()
    if success:
        print("\n🎉 Part 1 completed successfully!")
        print("\n📋 Part 1 Summary:")
        print("✅ ValidationThresholdEnv (Gymnasium environment)")
        print("✅ State-action-reward framework")
        print("✅ Data preprocessing utilities")
        print("✅ Execution safeguards")
        print("\n🚀 Ready for Part 2: PPO training and integration!")
    else:
        print("\n❌ Part 1 tests failed")
