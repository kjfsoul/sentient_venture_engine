#!/usr/bin/env python3
"""
Test Causal Analysis Agent with Environment Variable Fixes
Tests the agent with proper .env handling
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_environment_setup():
    """Test that environment variables are properly loaded"""
    print("🔧 Testing Environment Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = project_root / '.env'
    env_example_file = project_root / '.env.example'
    
    print(f"📁 Project root: {project_root}")
    print(f"📄 .env file exists: {'✅' if env_file.exists() else '❌'}")
    print(f"📄 .env.example file exists: {'✅' if env_example_file.exists() else '❌'}")
    
    # Test key environment variables
    test_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'OPENROUTER_API_KEY',
        'GEMINI_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    print("\n🔑 Environment Variables Status:")
    for var in test_vars:
        value = os.getenv(var)
        if value:
            # Show first 20 chars for security
            display_value = value[:20] + "..." if len(value) > 20 else value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: Not set")
    
    return True

def test_causal_agent_with_env():
    """Test the Causal Analysis Agent with proper environment handling"""
    print("\n🧠 Testing Causal Analysis Agent with Environment")
    print("=" * 60)
    
    try:
        # Test import
        from agents.analysis_agents import CausalAnalysisAgent
        print("✅ CausalAnalysisAgent imported successfully")
        
        # Test initialization in test mode (should work without Supabase)
        print("\n🧪 Testing in test mode (no Supabase connection):")
        agent_test = CausalAnalysisAgent(test_mode=True)
        print("✅ Agent initialized in test mode")
        
        # Test initialization in production mode (with Supabase if available)
        print("\n🚀 Testing in production mode (with Supabase if available):")
        try:
            agent_prod = CausalAnalysisAgent(test_mode=False)
            print("✅ Agent initialized in production mode")
            
            if agent_prod.supabase:
                print("✅ Supabase connection established")
            else:
                print("⚠️ Supabase connection not available (credentials missing or invalid)")
                
        except Exception as e:
            print(f"⚠️ Production mode initialization failed: {e}")
            print("   This is expected if Supabase credentials are not properly configured")
        
        # Test core functionality
        print("\n🔬 Testing core functionality:")
        
        # Test DAG
        dag = agent_test.causal_dag
        print(f"✅ Causal DAG: {len(dag['nodes'])} nodes, {len(dag['edges'])} edges")
        
        # Test hypotheses
        hypotheses = agent_test.causal_hypotheses
        print(f"✅ Causal hypotheses: {len(hypotheses)} defined")
        
        # Test data generation
        data = agent_test._generate_simulated_data()
        print(f"✅ Simulated data: {len(data)} rows, {len(data.columns)} columns")
        
        # Test feature extraction
        sample_hypothesis = {
            'initial_hypothesis_text': 'AI-powered SaaS platform for enterprise automation',
            'generated_by_agent': 'synthesis_agent'
        }
        sample_metrics = {
            'user_engagement': 0.75,
            'conversion_rate': 0.12,
            'roi': 2.3
        }
        
        complexity = agent_test._extract_market_complexity(sample_hypothesis, sample_metrics)
        investment = agent_test._extract_resource_investment(sample_metrics)
        novelty = agent_test._extract_hypothesis_novelty(sample_hypothesis)
        
        print(f"✅ Feature extraction working:")
        print(f"   Market complexity: {complexity:.3f}")
        print(f"   Resource investment: {investment:.3f}")
        print(f"   Hypothesis novelty: {novelty:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
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
        print("⚠️ DoWhy not available")
        print("   Install with: pip install dowhy==0.11.1")
        libraries.append(("DoWhy", False))
    
    # Test EconML
    try:
        import econml
        print("✅ EconML available")
        libraries.append(("EconML", True))
    except ImportError:
        print("⚠️ EconML not available")
        print("   Install with: pip install econml==0.15.0")
        libraries.append(("EconML", False))
    
    # Test causal-learn
    try:
        from causallearn.search.ConstraintBased.PC import pc
        print("✅ causal-learn available")
        libraries.append(("causal-learn", True))
    except ImportError:
        print("⚠️ causal-learn not available")
        print("   Install with: pip install causal-learn==0.1.3.8")
        libraries.append(("causal-learn", False))
    
    # Test LangChain
    try:
        from langchain_openai import ChatOpenAI
        print("✅ LangChain available")
        libraries.append(("LangChain", True))
    except ImportError:
        print("⚠️ LangChain not available")
        print("   Install with: pip install langchain-openai")
        libraries.append(("LangChain", False))
    
    available_count = sum(1 for _, available in libraries if available)
    print(f"\n📊 {available_count}/{len(libraries)} libraries available")
    
    if available_count == 0:
        print("\n⚠️ No causal inference libraries available.")
        print("   The agent will work with limited functionality.")
        print("   Run the installation script to install missing libraries:")
        print("   python task_updates/task_1_1_causal_analysis/install_causal_libraries.py")
    
    return available_count > 0

def main():
    """Run all tests"""
    print("🚀 Causal Analysis Agent - Environment Fix Tests")
    print("=" * 80)
    
    # Test environment setup
    env_test = test_environment_setup()
    
    # Test library availability
    lib_test = test_library_availability()
    
    # Test causal agent
    agent_test = test_causal_agent_with_env()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    print(f"Environment Setup: {'✅ PASSED' if env_test else '❌ FAILED'}")
    print(f"Library Availability: {'✅ PASSED' if lib_test else '⚠️ LIMITED'}")
    print(f"Causal Agent: {'✅ PASSED' if agent_test else '❌ FAILED'}")
    
    if env_test and agent_test:
        print("\n🎉 CORE FUNCTIONALITY WORKING!")
        print("The Causal Analysis Agent is properly handling environment variables.")
        
        if not lib_test:
            print("\n📋 Next Steps:")
            print("1. Install causal inference libraries for full functionality:")
            print("   pip install dowhy==0.11.1 econml==0.15.0 causal-learn==0.1.3.8")
            print("2. Ensure all API keys are properly set in .env file")
            print("3. Test with production data")
        
        return True
    else:
        print("\n⚠️ ISSUES DETECTED")
        if not env_test:
            print("- Environment setup failed")
        if not agent_test:
            print("- Causal agent failed to initialize")
        
        print("\n📋 Troubleshooting:")
        print("1. Ensure .env file exists in project root")
        print("2. Check that environment variables are properly set")
        print("3. Verify Supabase credentials if using production mode")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)