#!/usr/bin/env python3
"""
Installation script for causal inference libraries required for Task 1.3
Installs: DoWhy, EconML, causal-learn, and their dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def install_causal_libraries():
    """Install causal inference libraries"""
    print("🧠 Installing Causal Inference Libraries for SVE Task 1.3")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Activate conda environment if available
    conda_env = "/Users/kfitz/opt/anaconda3/bin/activate"
    if os.path.exists(conda_env):
        activate_cmd = f"source {conda_env} sve_env && "
    else:
        activate_cmd = ""
    
    # Libraries to install
    libraries = [
        {
            "name": "DoWhy",
            "package": "dowhy",
            "description": "Causal inference library with graphical models"
        },
        {
            "name": "EconML", 
            "package": "econml",
            "description": "Machine learning based causal inference"
        },
        {
            "name": "causal-learn",
            "package": "causal-learn",
            "description": "Causal discovery algorithms"
        },
        {
            "name": "NetworkX",
            "package": "networkx",
            "description": "Graph manipulation library (dependency)"
        },
        {
            "name": "scikit-learn",
            "package": "scikit-learn>=1.0.0",
            "description": "Machine learning library (dependency)"
        },
        {
            "name": "scipy",
            "package": "scipy>=1.7.0",
            "description": "Scientific computing library (dependency)"
        },
        {
            "name": "statsmodels",
            "package": "statsmodels",
            "description": "Statistical modeling library (dependency)"
        }
    ]
    
    success_count = 0
    total_count = len(libraries)
    
    for lib in libraries:
        print(f"\n📦 Installing {lib['name']}: {lib['description']}")
        
        # Try pip install
        command = f"{activate_cmd}pip install {lib['package']}"
        success = run_command(command, f"Installing {lib['package']}")
        
        if success:
            success_count += 1
        else:
            # Try conda install as fallback for some packages
            if lib['package'] in ['networkx', 'scikit-learn', 'scipy', 'statsmodels']:
                print(f"   Trying conda install as fallback...")
                conda_command = f"{activate_cmd}conda install -c conda-forge {lib['package'].split('>=')[0]} -y"
                conda_success = run_command(conda_command, f"Conda installing {lib['package']}")
                if conda_success:
                    success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All libraries installed successfully!")
        return True
    elif success_count >= 3:  # Core libraries
        print(f"\n⚠️ Most libraries installed. Some optional dependencies may be missing.")
        return True
    else:
        print(f"\n❌ Installation failed. Please install libraries manually.")
        return False

def verify_installation():
    """Verify that libraries are properly installed"""
    print(f"\n🔍 Verifying installation...")
    
    test_imports = [
        ("dowhy", "DoWhy"),
        ("econml", "EconML"), 
        ("causallearn", "causal-learn"),
        ("networkx", "NetworkX"),
        ("sklearn", "scikit-learn"),
        ("scipy", "SciPy"),
        ("statsmodels", "statsmodels")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {name} import failed: {e}")
    
    print(f"\n📊 Verification Summary:")
    print(f"   ✅ Working: {success_count}/{len(test_imports)}")
    
    return success_count >= 3  # Require at least core libraries

def main():
    """Main installation process"""
    print("🚀 Starting Causal Inference Library Installation")
    print("=" * 70)
    
    # Step 1: Install libraries
    install_success = install_causal_libraries()
    
    # Step 2: Verify installation
    if install_success:
        verify_success = verify_installation()
        
        if verify_success:
            print(f"\n✅ Installation completed successfully!")
            print(f"\n📋 Next steps:")
            print(f"   1. Run: python agents/causal_analysis_agent.py")
            print(f"   2. Test causal analysis with real data")
            print(f"   3. Integrate with SVE validation pipeline")
            return True
        else:
            print(f"\n⚠️ Installation completed with some issues.")
            print(f"   Some libraries may not be working correctly.")
            return False
    else:
        print(f"\n❌ Installation failed.")
        print(f"\n🔧 Manual installation commands:")
        print(f"   pip install dowhy econml causal-learn networkx scikit-learn scipy statsmodels")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
