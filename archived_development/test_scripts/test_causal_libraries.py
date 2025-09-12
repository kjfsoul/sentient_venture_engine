#!/usr/bin/env python3
"""
Quick test script for causal inference libraries
"""

def test_libraries():
    print("Testing causal inference libraries...")
    
    # Test DoWhy
    try:
        import dowhy
        from dowhy import CausalModel
        print("✅ DoWhy: OK")
    except ImportError:
        print("❌ DoWhy: FAILED")
    
    # Test EconML
    try:
        import econml
        from econml.dml import LinearDML
        print("✅ EconML: OK")
    except ImportError:
        print("❌ EconML: FAILED")
    
    # Test causal-learn
    try:
        from causallearn.search.ConstraintBased.PC import pc
        print("✅ causal-learn: OK")
    except ImportError:
        print("❌ causal-learn: FAILED")
    
    print("Library test completed!")

if __name__ == "__main__":
    test_libraries()
