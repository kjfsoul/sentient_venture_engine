#!/usr/bin/env python3
"""
Simple OpenRouter Cost Estimator
Quick cost analysis for your existing OpenRouter account
"""

def estimate_openrouter_costs():
    """Estimate costs for CrewAI workflows on OpenRouter"""
    
    print("💰 OPENROUTER COST ANALYSIS FOR YOUR ACCOUNT")
    print("=" * 60)
    
    # Model costs per 1M tokens (input/output)
    models = {
        "FREE MODELS": {
            "meta-llama/llama-3.1-8b-instruct": (0.0, 0.0),
            "mistralai/mistral-7b-instruct": (0.0, 0.0),
        },
        "ULTRA LOW COST": {
            "google/gemini-flash-1.5": (0.075, 0.30),
            "anthropic/claude-3-haiku": (0.25, 1.25),
            "openai/gpt-4o-mini": (0.15, 0.60),
        },
        "MODERATE COST": {
            "anthropic/claude-3-5-sonnet": (3.0, 15.0),
            "openai/gpt-4o": (2.5, 10.0),
        }
    }
    
    # Typical CrewAI workflow usage
    workflow_tokens = {
        "input": 15000,   # ~15k tokens input per workflow
        "output": 8000    # ~8k tokens output per workflow
    }
    
    print("🔍 COST PER CREWAI WORKFLOW:")
    print()
    
    for tier, tier_models in models.items():
        print(f"📋 {tier}:")
        for model, (input_cost, output_cost) in tier_models.items():
            # Calculate cost per workflow
            total_cost = (workflow_tokens["input"] * input_cost / 1_000_000) + (workflow_tokens["output"] * output_cost / 1_000_000)
            
            if total_cost == 0:
                print(f"   {model}: FREE! 🎉")
            else:
                print(f"   {model}: ${total_cost:.4f}")
        print()
    
    # Budget recommendations
    print("💡 BUDGET RECOMMENDATIONS:")
    print("=" * 40)
    print("✅ START WITH FREE MODELS:")
    print("   • meta-llama/llama-3.1-8b-instruct")
    print("   • Cost: $0.00 per workflow")
    print("   • Great for development and testing")
    print()
    print("📈 UPGRADE TO LOW-COST WHEN NEEDED:")
    print("   • google/gemini-flash-1.5")
    print("   • Cost: ~$0.004 per workflow")
    print("   • $5 budget = ~1,250 workflows")
    print()
    print("🚀 PRODUCTION RECOMMENDATION:")
    print("   Daily Budget: $5-10")
    print("   Monthly Budget: $50-150")
    print("   Expected Usage: 100-500 workflows/month")
    
    return True

if __name__ == "__main__":
    estimate_openrouter_costs()
