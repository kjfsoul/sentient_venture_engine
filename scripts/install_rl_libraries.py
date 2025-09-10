#!/usr/bin/env python3
"""
Installation script for RL libraries required for Task 1.4: Dynamic Threshold Adjustment
Installs: Stable Baselines3, Gymnasium, and dependencies
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

def install_rl_libraries():
    """Install RL libraries for Task 1.4"""
    print("🤖 Installing RL Libraries for SVE Task 1.4")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Libraries to install
    libraries = [
        {
            "name": "Gymnasium", 
            "package": "gymnasium[classic_control]",
            "description": "OpenAI Gym replacement for RL environments"
        },
        {
            "name": "Stable Baselines3",
            "package": "stable-baselines3[extra]",
            "description": "RL algorithms (PPO, A2C, SAC, etc.)"
        },
        {
            "name": "PyTorch",
            "package": "torch",
            "description": "Deep learning framework (SB3 dependency)"
        },
        {
            "name": "NumPy",
            "package": "numpy>=1.21.0",
            "description": "Numerical computing library"
        },
        {
            "name": "Pandas",
            "package": "pandas>=1.3.0", 
            "description": "Data manipulation library"
        },
        {
            "name": "Matplotlib",
            "package": "matplotlib",
            "description": "Plotting library for training visualizations"
        },
        {
            "name": "TensorBoard",
            "package": "tensorboard",
            "description": "Training monitoring and visualization"
        }
    ]
    
    success_count = 0
    total_count = len(libraries)
    
    for lib in libraries:
        print(f"\n📦 Installing {lib['name']}: {lib['description']}")
        
        # Try pip install
        command = f"pip install {lib['package']}"
        success = run_command(command, f"Installing {lib['package']}")
        
        if success:
            success_count += 1
        else:
            # Try with --upgrade flag
            print(f"   Retrying with --upgrade...")
            upgrade_command = f"pip install --upgrade {lib['package']}"
            upgrade_success = run_command(upgrade_command, f"Upgrading {lib['package']}")
            if upgrade_success:
                success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count >= 5:  # Core libraries
        print(f"\n🎉 Core RL libraries installed successfully!")
        return True
    else:
        print(f"\n❌ Installation failed. Please install libraries manually.")
        return False

def verify_installation():
    """Verify that RL libraries are properly installed"""
    print(f"\n🔍 Verifying installation...")
    
    test_imports = [
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable Baselines3"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard")
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
    
    return success_count >= 5  # Require core libraries

def main():
    """Main installation process"""
    print("🚀 Starting RL Library Installation for Task 1.4")
    print("=" * 70)
    
    # Step 1: Install libraries
    install_success = install_rl_libraries()
    
    # Step 2: Verify installation
    if install_success:
        verify_success = verify_installation()
        
        if verify_success:
            print(f"\n✅ Installation completed successfully!")
            print(f"\n📋 Next steps:")
            print(f"   1. Run: python agents/rl_threshold_optimizer_part1.py")
            print(f"   2. Test RL environment functionality")
            print(f"   3. Proceed to Part 2: PPO training implementation")
            return True
        else:
            print(f"\n⚠️ Installation completed with some issues.")
            return False
    else:
        print(f"\n❌ Installation failed.")
        print(f"\n🔧 Manual installation commands:")
        print(f"   pip install gymnasium stable-baselines3 torch numpy pandas matplotlib tensorboard")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
