#!/usr/bin/env python3
"""
Quick validation of the Causal Analysis Agent core functionality
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def quick_test():
    """Quick test of core functionality"""
    print("🧠 Quick Causal Analysis Agent Validation")
    
    try:
        # Test import
        from agents.analysis_agents import CausalAnalysisAgent
        print("✅ Import successful")
        
        # Test initialization in test mode
        agent = CausalAnalysisAgent(test_mode=True)
        print("✅ Agent initialized")
        
        # Test DAG
        dag = agent.causal_dag
        print(f"✅ DAG: {len(dag['nodes'])} nodes")
        
        # Test hypotheses
        hypotheses = agent.causal_hypotheses
        print(f"✅ Hypotheses: {len(hypotheses)} defined")
        
        # Test data generation
        data = agent._generate_simulated_data()
        print(f"✅ Data: {len(data)} rows generated")
        
        print("\n🎉 CORE FUNCTIONALITY WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_test()