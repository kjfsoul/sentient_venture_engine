#!/usr/bin/env python3
"""
Install Causal Inference Libraries for SVE Project
Installs DoWhy, EconML, and causal-learn with proper dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_causal_libraries():
    """Install all required causal inference libraries"""
    print("🚀 Installing Causal Inference Libraries for SVE Project")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required for causal inference libraries")
        return False
    
    # Install core dependencies first
    core_deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6",
        "matplotlib>=3.5.0"
    ]
    
    print("\n📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")
    
    # Install causal inference libraries
    causal_libraries = [
        ("dowhy==0.11.1", "DoWhy - Unified causal inference framework"),
        ("econml==0.15.0", "EconML - Machine learning based causal inference"),
        ("causal-learn==0.1.3.8", "causal-learn - Causal discovery algorithms"),
        ("graphviz==0.20.1", "Graphviz - Graph visualization for causal diagrams")
    ]
    
    print("\n🧠 Installing causal inference libraries...")
    success_count = 0
    
    for library, description in causal_libraries:
        if run_command(f"pip install '{library}'", f"Installing {description}"):
            success_count += 1
        else:
            print(f"⚠️ Failed to install {library}")
    
    # Install optional dependencies for better performance
    optional_deps = [
        ("torch", "PyTorch for advanced ML models in EconML"),
        ("lightgbm", "LightGBM for gradient boosting in causal inference"),
        ("xgboost", "XGBoost for ensemble methods")
    ]
    
    print("\n⚡ Installing optional performance dependencies...")
    for dep, description in optional_deps:
        run_command(f"pip install {dep}", f"Installing {description}")
    
    # Test installations
    print("\n🧪 Testing library installations...")
    test_results = []
    
    # Test DoWhy
    try:
        import dowhy
        print(f"✅ DoWhy {dowhy.__version__} installed successfully")
        test_results.append(("DoWhy", True))
    except ImportError as e:
        print(f"❌ DoWhy import failed: {e}")
        test_results.append(("DoWhy", False))
    
    # Test EconML
    try:
        import econml
        print(f"✅ EconML {econml.__version__} installed successfully")
        test_results.append(("EconML", True))
    except ImportError as e:
        print(f"❌ EconML import failed: {e}")
        test_results.append(("EconML", False))
    
    # Test causal-learn
    try:
        import causallearn
        print(f"✅ causal-learn installed successfully")
        test_results.append(("causal-learn", True))
    except ImportError as e:
        print(f"❌ causal-learn import failed: {e}")
        test_results.append(("causal-learn", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    successful_installs = sum(1 for _, success in test_results if success)
    total_libraries = len(test_results)
    
    for library, success in test_results:
        status = "✅ INSTALLED" if success else "❌ FAILED"
        print(f"{library}: {status}")
    
    print(f"\nOverall: {successful_installs}/{total_libraries} libraries installed successfully")
    
    if successful_installs == total_libraries:
        print("\n🎉 ALL CAUSAL INFERENCE LIBRARIES INSTALLED SUCCESSFULLY!")
        print("The Causal Analysis Agent is ready to use.")
        return True
    else:
        print(f"\n⚠️ {total_libraries - successful_installs} libraries failed to install.")
        print("The Causal Analysis Agent will work with reduced functionality.")
        return False

def create_test_script():
    """Create a simple test script to verify installations"""
    test_script = '''#!/usr/bin/env python3
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
'''
    
    with open("test_causal_libraries.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_causal_libraries.py for quick testing")

if __name__ == "__main__":
    success = install_causal_libraries()
    create_test_script()
    
    if success:
        print("\n🚀 Ready to run causal analysis!")
        print("Next steps:")
        print("1. Run: python test_causal_libraries.py")
        print("2. Run: python test_causal_analysis_comprehensive.py")
        print("3. Use the CausalAnalysisAgent in your SVE workflows")
    else:
        print("\n⚠️ Some libraries failed to install.")
        print("You may need to install them manually or check system dependencies.")
    
    sys.exit(0 if success else 1)