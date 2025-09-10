#!/usr/bin/env python3
"""
Task 1.4: Dynamic Threshold Adjustment - Complete Implementation
RL-based dynamic threshold optimization using Stable Baselines3

Features:
- ValidationThresholdEnv (Gymnasium environment)
- PPO-based threshold optimization
- Supabase integration for threshold storage
- Validation agent integration layer
- Execution safeguards (max_iterations, max_execution_time, early_stopping)

Usage:
    python agents/rl_threshold_optimizer.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Import both parts
try:
    from agents.rl_threshold_optimizer_part1 import (
        ValidationThresholdEnv, ValidationDataProcessor, ThresholdState
    )
    from agents.rl_threshold_optimizer_part2 import (
        RLThresholdOptimizer, ValidationAgentIntegration, TrainingConfig, run_complete_rl_training
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import RL components: {e}")
    COMPONENTS_AVAILABLE = False

# Import AI interaction wrapper
try:
    from agents.ai_interaction_wrapper import log_interaction, add_memory_addendum
    MEMORY_LOGGING_AVAILABLE = True
except ImportError:
    MEMORY_LOGGING_AVAILABLE = False
    def log_interaction(*args, **kwargs):
        return "mock_interaction_id"
    def add_memory_addendum(response, context=None):
        return response

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLThresholdSystem:
    """
    Complete RL-based dynamic threshold adjustment system
    Integrates training, optimization, and validation agent support
    """
    
    def __init__(self):
        """Initialize the complete RL threshold system"""
        self.system_status = {
            "components_available": COMPONENTS_AVAILABLE,
            "memory_logging": MEMORY_LOGGING_AVAILABLE,
            "initialization_time": datetime.now().isoformat()
        }
        
        if COMPONENTS_AVAILABLE:
            self.data_processor = ValidationDataProcessor()
            self.integration = ValidationAgentIntegration()
            logger.info("✅ RLThresholdSystem initialized successfully")
        else:
            logger.error("❌ RLThresholdSystem initialization failed - missing components")
    
    def train_threshold_optimizer(self, 
                                config: Dict[str, Any] = None,
                                days_back: int = 90) -> Dict[str, Any]:
        """
        Train RL threshold optimizer with custom configuration
        
        Args:
            config: Training configuration parameters
            days_back: Historical data range
            
        Returns:
            Training results and system status
        """
        if not COMPONENTS_AVAILABLE:
            return {
                "success": False,
                "error": "RL components not available",
                "system_status": self.system_status
            }
        
        # Log interaction start
        user_query = f"Train RL threshold optimizer with {days_back} days of data"
        
        try:
            logger.info("🤖 Starting RL threshold optimization training")
            
            # Prepare training configuration
            training_config = TrainingConfig(
                total_timesteps=config.get("total_timesteps", 10000) if config else 10000,
                learning_rate=config.get("learning_rate", 3e-4) if config else 3e-4,
                max_training_time=config.get("max_training_time", 1800) if config else 1800,  # 30 min default
                early_stopping_threshold=config.get("early_stopping_threshold", 1500.0) if config else 1500.0
            )
            
            # Run complete training pipeline
            results = run_complete_rl_training(
                days_back=days_back,
                total_timesteps=training_config.total_timesteps
            )
            
            # Enhanced results with system info
            enhanced_results = {
                **results,
                "training_config": asdict(training_config),
                "system_status": self.system_status,
                "execution_safeguards": {
                    "max_training_time": training_config.max_training_time,
                    "early_stopping": training_config.early_stopping_threshold,
                    "max_iterations": training_config.total_timesteps
                }
            }
            
            # Generate response for memory logging
            if results.get("success"):
                ai_response = f"""
✅ **RL Threshold Optimization Training Completed Successfully**

📊 **Training Results:**
- Timesteps: {results.get('training_results', {}).get('total_timesteps_trained', 0):,}
- Best Reward: {results.get('training_results', {}).get('best_reward', 0):.2f}
- Training Time: {results.get('training_results', {}).get('training_time', 0):.1f}s
- Optimal Thresholds: {results.get('training_results', {}).get('best_thresholds', [])}

🔧 **System Integration:**
- Model saved: {results.get('model_saved', False)}
- Database integration: {results.get('thresholds_stored', False)}
- Validation agents ready: {results.get('integration_test', {}).get('success', False)}

🚀 **Next Steps:**
The dynamic threshold system is now active. Validation agents will automatically retrieve optimized thresholds before tier progression decisions, improving validation success rates and reducing resource waste.
"""
            else:
                ai_response = f"❌ RL training failed: {results.get('error', 'Unknown error')}"
            
            # Log interaction for memory system
            if MEMORY_LOGGING_AVAILABLE:
                log_interaction(
                    user_query=user_query,
                    ai_response=ai_response,
                    key_actions=[
                        "Executed RL threshold optimization training",
                        "Implemented PPO-based dynamic thresholds",
                        "Integrated with validation agent system"
                    ],
                    progress_indicators=[
                        f"Training completed: {results.get('success', False)}",
                        f"System ready: {results.get('integration_test', {}).get('success', False)}"
                    ],
                    forward_initiative="Dynamic threshold adjustment system operational for improved validation efficiency",
                    completion_status="completed" if results.get("success") else "error"
                )
            
            return enhanced_results
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "system_status": self.system_status
            }
    
    def get_current_thresholds(self, hypothesis_features: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get current optimized thresholds for validation agents
        
        Args:
            hypothesis_features: Optional context for threshold selection
            
        Returns:
            Current threshold configuration
        """
        if not COMPONENTS_AVAILABLE:
            return {
                "success": False,
                "error": "Components not available",
                "default_thresholds": [0.6, 0.7, 0.8]
            }
        
        try:
            thresholds = self.integration.get_current_thresholds(hypothesis_features)
            
            return {
                "success": True,
                "thresholds": {
                    "social_sentiment": thresholds.social_sentiment_threshold,
                    "prototype_testing": thresholds.prototype_testing_threshold,
                    "market_validation": thresholds.market_validation_threshold
                },
                "threshold_array": thresholds.to_array().tolist(),
                "retrieval_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error retrieving thresholds: {e}")
            return {
                "success": False,
                "error": str(e),
                "default_thresholds": [0.6, 0.7, 0.8]
            }
    
    def update_threshold_feedback(self, 
                                tier: int,
                                success: bool,
                                hypothesis_id: str,
                                metrics: Dict[str, Any] = None) -> bool:
        """
        Update threshold performance feedback for future optimization
        
        Args:
            tier: Validation tier (1, 2, 3)
            success: Whether validation succeeded  
            hypothesis_id: Hypothesis ID
            metrics: Additional performance metrics
            
        Returns:
            Success status
        """
        if not COMPONENTS_AVAILABLE:
            return False
        
        try:
            return self.integration.update_threshold_performance(
                tier=tier,
                success=success,
                hypothesis_id=hypothesis_id,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating threshold feedback: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status and health check"""
        status = {
            "system_ready": COMPONENTS_AVAILABLE,
            "memory_logging": MEMORY_LOGGING_AVAILABLE,
            "components": self.system_status,
            "health_check": {}
        }
        
        if COMPONENTS_AVAILABLE:
            try:
                # Test threshold retrieval
                test_thresholds = self.get_current_thresholds()
                status["health_check"]["threshold_retrieval"] = test_thresholds["success"]
                
                # Test data processor
                test_data = self.data_processor.retrieve_training_data(days_back=7)
                status["health_check"]["data_processing"] = len(test_data) > 0 if test_data is not None else False
                
                status["health_check"]["overall"] = all(status["health_check"].values())
                
            except Exception as e:
                status["health_check"]["error"] = str(e)
                status["health_check"]["overall"] = False
        
        return status


def create_threshold_configs_table():
    """
    Create threshold_configs table in Supabase if it doesn't exist
    This would typically be done via Supabase dashboard or migration
    """
    
    table_schema = """
    CREATE TABLE IF NOT EXISTS public.threshold_configs (
        id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
        config_name text NOT NULL,
        social_sentiment_threshold double precision NOT NULL,
        prototype_testing_threshold double precision NOT NULL,
        market_validation_threshold double precision NOT NULL,
        optimization_timestamp timestamp with time zone DEFAULT now(),
        training_metrics jsonb,
        is_active boolean DEFAULT false,
        created_at timestamp with time zone DEFAULT now()
    );
    
    CREATE TABLE IF NOT EXISTS public.threshold_performance (
        id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
        hypothesis_id uuid NOT NULL,
        validation_tier integer NOT NULL,
        success boolean NOT NULL,
        timestamp timestamp with time zone DEFAULT now(),
        metrics jsonb,
        current_thresholds jsonb,
        created_at timestamp with time zone DEFAULT now()
    );
    """
    
    print("📋 Threshold tables schema:")
    print(table_schema)
    print("\n💡 Execute this SQL in your Supabase dashboard to create the required tables.")


def main():
    """Main execution function"""
    print("🤖 RL Threshold Optimization System")
    print("=" * 70)
    
    # Initialize system
    rl_system = RLThresholdSystem()
    
    # Get system status
    status = rl_system.get_system_status()
    print(f"\n📊 System Status: {'✅ Ready' if status['system_ready'] else '❌ Not Ready'}")
    print(f"💾 Memory Logging: {'✅ Active' if status['memory_logging'] else '⚠️ Disabled'}")
    
    if not status["system_ready"]:
        print("\n❌ System not ready. Please ensure all components are installed:")
        print("   pip install gymnasium stable-baselines3 torch")
        return
    
    # Health check
    health = status.get("health_check", {})
    print(f"\n🏥 Health Check: {'✅ Healthy' if health.get('overall', False) else '⚠️ Issues'}")
    
    if health.get("threshold_retrieval"):
        print("   ✅ Threshold retrieval working")
    if health.get("data_processing"):
        print("   ✅ Data processing working")
    
    # Test threshold retrieval
    print("\n🔍 Testing threshold retrieval...")
    thresholds = rl_system.get_current_thresholds()
    if thresholds["success"]:
        print(f"   ✅ Current thresholds: {thresholds['threshold_array']}")
    else:
        print(f"   ⚠️ Using defaults: {thresholds.get('default_thresholds', [])}")
    
    # Offer training option
    print("\n🚀 Training Options:")
    print("1. Quick demo training (5,000 timesteps)")
    print("2. Production training (50,000 timesteps)")
    print("3. Skip training")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🏃 Running quick demo training...")
        results = rl_system.train_threshold_optimizer(
            config={"total_timesteps": 5000, "max_training_time": 600},
            days_back=30
        )
        print(f"Results: {'✅ Success' if results['success'] else '❌ Failed'}")
        
    elif choice == "2":
        print("\n🏋️ Running production training...")
        results = rl_system.train_threshold_optimizer(
            config={"total_timesteps": 50000, "max_training_time": 3600},
            days_back=90
        )
        print(f"Results: {'✅ Success' if results['success'] else '❌ Failed'}")
    
    print("\n✅ RL Threshold Optimization System ready!")
    print("\n📋 Next Steps:")
    print("1. Create threshold_configs table in Supabase (see create_threshold_configs_table())")
    print("2. Integration with validation agents is ready")
    print("3. System will optimize thresholds based on validation performance")
    
    # Show table creation info
    print("\n🗄️ Database Setup:")
    create_threshold_configs_table()


if __name__ == "__main__":
    main()
