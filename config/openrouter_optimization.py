#!/usr/bin/env python3
"""
OpenRouter Cost Optimization Configuration
Maximize efficiency with your existing OpenRouter credits

This configuration optimizes token usage and selects the most cost-effective models
while maintaining high-quality outputs for CrewAI integration.
"""

import os
from typing import Dict, List, Tuple

class OpenRouterOptimizer:
    """Optimize OpenRouter usage for cost-effective CrewAI operations"""
    
    def __init__(self):
        # Model costs per 1M tokens (input/output) in USD
        self.model_costs = {
            # FREE MODELS (Best value - $0.00)
            "meta-llama/llama-3.1-8b-instruct": (0.000, 0.000),
            "mistralai/mistral-7b-instruct": (0.000, 0.000),
            "microsoft/phi-3-mini-128k-instruct": (0.000, 0.000),
            "google/gemma-7b-it": (0.000, 0.000),
            
            # ULTRA LOW COST MODELS ($0.20-0.50 per 1M tokens)
            "meta-llama/llama-3.1-8b-instruct:extended": (0.187, 0.750),
            "google/gemini-flash-1.5": (0.075, 0.300),
            "anthropic/claude-3-haiku": (0.250, 1.250),
            "openai/gpt-4o-mini": (0.150, 0.600),
            
            # MODERATE COST MODELS ($1-3 per 1M tokens)
            "anthropic/claude-3-5-sonnet": (3.000, 15.000),
            "openai/gpt-4o": (2.500, 10.000),
            "google/gemini-pro-1.5": (1.250, 5.000),
            
            # HIGH PERFORMANCE MODELS ($5+ per 1M tokens)
            "anthropic/claude-3-opus": (15.000, 75.000),
            "openai/gpt-4-turbo": (10.000, 30.000)
        }
        
        # Task complexity to model mapping
        self.task_model_mapping = {
            "simple": ["meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct"],
            "moderate": ["google/gemini-flash-1.5", "anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
            "complex": ["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "google/gemini-pro-1.5"],
            "critical": ["anthropic/claude-3-opus", "openai/gpt-4-turbo"]
        }
    
    def get_optimal_model_for_task(self, task_type: str, budget_tier: str = "free") -> str:
        """Get the optimal model for a specific task and budget"""
        
        if budget_tier == "free":
            # Always use free models first
            free_models = [model for model, cost in self.model_costs.items() 
                          if cost[0] == 0.000 and cost[1] == 0.000]
            if free_models:
                return free_models[0]  # Use best free model
        
        # Task-specific model selection
        task_complexity = self._determine_task_complexity(task_type)
        candidate_models = self.task_model_mapping.get(task_complexity, self.task_model_mapping["moderate"])
        
        # Return most cost-effective model for the complexity
        return candidate_models[0]
    
    def _determine_task_complexity(self, task_type: str) -> str:
        """Determine task complexity based on task type"""
        simple_tasks = ["basic_analysis", "data_extraction", "simple_classification"]
        moderate_tasks = ["market_analysis", "competitive_analysis", "hypothesis_generation"]
        complex_tasks = ["strategic_planning", "business_model_design", "comprehensive_synthesis"]
        critical_tasks = ["financial_modeling", "legal_analysis", "high_stakes_decisions"]
        
        if task_type in simple_tasks:
            return "simple"
        elif task_type in moderate_tasks:
            return "moderate"
        elif task_type in complex_tasks:
            return "complex"
        elif task_type in critical_tasks:
            return "critical"
        else:
            return "moderate"  # Default
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model and token usage"""
        if model not in self.model_costs:
            return 0.0  # Free if unknown
        
        input_cost, output_cost = self.model_costs[model]
        total_cost = (input_tokens * input_cost / 1_000_000) + (output_tokens * output_cost / 1_000_000)
        return total_cost
    
    def get_cost_optimized_config(self) -> Dict:
        """Get cost-optimized configuration for CrewAI"""
        return {
            "model_priority": [
                # Start with free models
                "meta-llama/llama-3.1-8b-instruct",
                "mistralai/mistral-7b-instruct",
                "microsoft/phi-3-mini-128k-instruct",
                # Fallback to ultra-low cost
                "google/gemini-flash-1.5",
                "anthropic/claude-3-haiku",
                "openai/gpt-4o-mini"
            ],
            "token_limits": {
                "free_models": 4000,      # Conservative for free models
                "low_cost": 8000,         # More generous for paid models
                "moderate_cost": 12000,   # Full capacity for moderate cost
                "high_cost": 16000        # Maximum for high-performance models
            },
            "retry_strategy": {
                "max_retries": 3,
                "backoff_factor": 2,
                "timeout": 60
            }
        }

def configure_openrouter_for_crewai():
    """Configure OpenRouter settings optimized for CrewAI workflows"""
    
    optimizer = OpenRouterOptimizer()
    config = optimizer.get_cost_optimized_config()
    
    # Set environment variables for optimal performance
    os.environ.setdefault("LLM_MAX_TOKENS", "4000")  # Conservative default
    os.environ.setdefault("LLM_TEMPERATURE", "0.7")   # Balanced creativity
    os.environ.setdefault("OPENROUTER_TIMEOUT", "60") # Reasonable timeout
    
    print("🔧 OpenRouter Optimization Configuration")
    print("=" * 50)
    print(f"💡 Primary Models: {', '.join(config['model_priority'][:3])}")
    print(f"🎯 Token Limit Strategy: {config['token_limits']['free_models']} (free) → {config['token_limits']['moderate_cost']} (paid)")
    print(f"🔄 Retry Strategy: {config['retry_strategy']['max_retries']} retries with backoff")
    
    return config

def estimate_crewai_workflow_cost():
    """Estimate cost for a complete CrewAI workflow"""
    
    optimizer = OpenRouterOptimizer()
    
    # Typical CrewAI workflow token usage
    workflow_stages = {
        "market_analysis": {"input": 3000, "output": 1500},
        "business_model_design": {"input": 2500, "output": 2000},
        "competitive_analysis": {"input": 3500, "output": 1800},
        "hypothesis_formulation": {"input": 4000, "output": 2500},
        "vetting_evaluation": {"input": 2000, "output": 800}
    }
    
    print("\n💰 CrewAI Workflow Cost Estimation")
    print("=" * 50)
    
    for tier, models in [("FREE", ["meta-llama/llama-3.1-8b-instruct"]),
                        ("LOW COST", ["google/gemini-flash-1.5"]),
                        ("MODERATE", ["anthropic/claude-3-5-sonnet"])]:
        
        total_cost = 0
        model = models[0]
        
        print(f"\n🏷️ {tier} TIER ({model}):")
        
        for stage, tokens in workflow_stages.items():
            stage_cost = optimizer.estimate_cost(model, tokens["input"], tokens["output"])
            total_cost += stage_cost
            print(f"   {stage}: ${stage_cost:.4f}")
        
        print(f"   📊 TOTAL: ${total_cost:.4f} per complete workflow")
        
        if total_cost == 0:
            print(f"   🎉 FREE TIER - No cost!")

if __name__ == "__main__":
    # Configure OpenRouter optimization
    config = configure_openrouter_for_crewai()
    
    # Show cost estimates
    estimate_crewai_workflow_cost()
    
    print(f"\n🚀 RECOMMENDATION:")
    print(f"   Use FREE models first: meta-llama/llama-3.1-8b-instruct")
    print(f"   Fallback to: google/gemini-flash-1.5 ($0.075/$0.30 per 1M tokens)")
    print(f"   Your existing OpenRouter credits should last for hundreds of workflows!")
