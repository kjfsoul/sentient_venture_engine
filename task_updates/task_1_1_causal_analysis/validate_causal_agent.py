#!/usr/bin/env python3
"""
Simple validation script for Causal Analysis Agent
"""

import sys
import os

def test_basic_functionality():
    """Test basic functionality of the Causal Analysis Agent"""
    print("🧠 Testing Causal Analysis Agent Basic Functionality")
    print("=" * 60)
    
    try:
        # Test import
        from agents.analysis_agents import CausalAnalysisAgent
        print("✅ CausalAnalysisAgent imported successfully")
        
        # Initialize agent in test mode
        agent = CausalAnalysisAgent(test_mode=True)
        print("✅ Agent initialized in test mode")
        
        # Test causal DAG
        dag = agent.causal_dag
        print(f"✅ Causal DAG defined with {len(dag['nodes'])} nodes and {len(dag['edges'])} edges")
        
        # Test causal hypotheses
        hypotheses = agent.causal_hypotheses
        print(f"✅ {len(hypotheses)} causal hypotheses defined")
        
        # Test data generation
        data = agent._generate_simulated_data()
        print(f"✅ Simulated data generated: {len(data)} rows, {len(data.columns)} columns")
        
        # Test feature extraction
        sample_hypothesis = {'initial_hypothesis_text': 'AI-powered SaaS platform'}
        sample_metrics = {'user_engagement': 0.7}
        
        complexity = agent._extract_market_complexity(sample_hypothesis, sample_metrics)
        print(f"✅ Feature extraction working: market_complexity = {complexity:.3f}")
        
        print("\n🎉 Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_library_availability():
    """Test availability of causal inference libraries"""
    print("\n📚 Testing Causal Inference Library Availability")
    print("=" * 60)
    
    libraries = []
    
    # Test DoWhy
    try:
        import dowhy
        print("✅ DoWhy available")
        libraries.append(("DoWhy", True))
    except ImportError:
        print("⚠️ DoWhy not available - install with: pip install dowhy")
        libraries.append(("DoWhy", False))
    
    # Test EconML
    try:
        import econml
        print("✅ EconML available")
        libraries.append(("EconML", True))
    except ImportError:
        print("⚠️ EconML not available - install with: pip install econml")
        libraries.append(("EconML", False))
    
    # Test causal-learn
    try:
        from causallearn.search.ConstraintBased.PC import pc
        print("✅ causal-learn available")
        libraries.append(("causal-learn", True))
    except ImportError:
        print("⚠️ causal-learn not available - install with: pip install causal-learn")
        libraries.append(("causal-learn", False))
    
    available_count = sum(1 for _, available in libraries if available)
    print(f"\n📊 {available_count}/{len(libraries)} causal inference libraries available")
    
    return available_count > 0

def main():
    """Run validation tests"""
    print("🚀 Causal Analysis Agent Validation")
    print("=" * 80)
    
    # Test basic functionality
    basic_test = test_basic_functionality()
    
    # Test library availability
    library_test = test_library_availability()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"Basic Functionality: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    print(f"Library Availability: {'✅ PASSED' if library_test else '❌ FAILED'}")
    
    if basic_test and library_test:
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("The Causal Analysis Agent is ready for use.")
        print("\nNext steps:")
        print("1. Install missing libraries if any: python install_causal_libraries.py")
        print("2. Run comprehensive tests: python test_causal_analysis_comprehensive.py")
        print("3. Integrate with SVE workflow")
        return True
    else:
        print("\n⚠️ VALIDATION ISSUES DETECTED")
        if not basic_test:
            print("- Basic functionality failed - check implementation")
        if not library_test:
            print("- No causal libraries available - run installation script")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)