#!/usr/bin/env python3
"""
BULLETPROOF CrewAI Integration Test & Execution
NEVER SETTLE FOR LESS - Guaranteed working CrewAI

This script validates and executes CrewAI integration with zero tolerance for failures.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_setup():
    """Test conda environment and package installation"""
    print("🔧 TESTING ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Test 1: Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version >= (3, 9):
        print("   ✅ Python version compatible")
    else:
        print("   ❌ Python version too old")
        return False
    
    # Test 2: CrewAI installation
    try:
        import crewai
        from crewai import Agent, Task, Crew
        print("   ✅ CrewAI installed and importable")
    except ImportError as e:
        print(f"   ❌ CrewAI import failed: {e}")
        return False
    
    # Test 3: LangChain installation
    try:
        from langchain_openai import ChatOpenAI
        print("   ✅ LangChain OpenAI installed")
    except ImportError as e:
        print(f"   ❌ LangChain OpenAI import failed: {e}")
        return False
    
    # Test 4: Other dependencies
    required_packages = ['supabase', 'pandas', 'numpy', 'pydantic']
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} installed")
        except ImportError:
            print(f"   ❌ {package} missing")
            return False
    
    return True

def test_llm_providers():
    """Test LLM provider functionality"""
    print("\n🤖 TESTING LLM PROVIDERS")
    print("=" * 60)
    
    try:
        from agents.bulletproof_llm_provider import get_bulletproof_llm, test_llm_providers
        
        # Run comprehensive LLM tests
        test_llm_providers()
        
        # Get working LLM
        llm = get_bulletproof_llm()
        print(f"\n✅ Bulletproof LLM provider working")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM provider test failed: {e}")
        return False

def test_crewai_basic_functionality():
    """Test basic CrewAI functionality"""
    print("\n🚀 TESTING CREWAI BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from crewai import Agent, Task, Crew
        from agents.bulletproof_llm_provider import get_bulletproof_llm
        
        # Get working LLM
        llm = get_bulletproof_llm()
        
        # Create a simple agent
        test_agent = Agent(
            role='Test Agent',
            goal='Test CrewAI functionality',
            backstory='A simple test agent to validate CrewAI integration.',
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=30
        )
        
        # Create a simple task
        test_task = Task(
            description="Say 'CrewAI is working perfectly' and nothing else.",
            agent=test_agent,
            expected_output="Confirmation that CrewAI is working"
        )
        
        # Create and run crew
        test_crew = Crew(
            agents=[test_agent],
            tasks=[test_task],
            verbose=False
        )
        
        print("   🔄 Running basic CrewAI test...")
        result = test_crew.kickoff()
        
        if result:
            print(f"   ✅ CrewAI execution successful")
            print(f"   📋 Result: {str(result)[:100]}...")
            return True
        else:
            print(f"   ❌ CrewAI execution returned no result")
            return False
            
    except Exception as e:
        print(f"   ❌ CrewAI basic test failed: {e}")
        return False

def test_synthesis_agents():
    """Test synthesis agents functionality"""
    print("\n🧠 TESTING SYNTHESIS AGENTS")
    print("=" * 60)
    
    try:
        from agents.synthesis_agents import MarketOpportunityAgent
        
        # Initialize agent
        agent = MarketOpportunityAgent()
        print("   ✅ MarketOpportunityAgent initialized")
        
        # Test market intelligence retrieval (should work with fallback data)
        market_data = agent.retrieve_market_intelligence(days_back=7)
        if market_data:
            print(f"   ✅ Market intelligence retrieval: {len(market_data)} records")
        else:
            print("   ❌ Market intelligence retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Synthesis agents test failed: {e}")
        return False

def test_vetting_agent():
    """Test vetting agent functionality"""
    print("\n🎯 TESTING VETTING AGENT")
    print("=" * 60)
    
    try:
        from agents.vetting_agent import VettingAgent, HypothesisVettingEngine
        
        # Test vetting engine
        engine = HypothesisVettingEngine()
        print("   ✅ HypothesisVettingEngine initialized")
        
        # Test vetting agent
        agent = VettingAgent()
        print("   ✅ VettingAgent initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Vetting agent test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔄 TESTING INTEGRATION WORKFLOW")
    print("=" * 60)
    
    try:
        from scripts.test_vetting_integration import main as test_vetting_main
        
        # Run the vetting integration test
        print("   🔄 Running vetting integration test...")
        success = test_vetting_main()
        
        if success:
            print("   ✅ Integration workflow test passed")
            return True
        else:
            print("   ❌ Integration workflow test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration workflow test failed: {e}")
        return False

def run_production_workflow():
    """Run the actual production workflow"""
    print("\n🚀 RUNNING PRODUCTION WORKFLOW")
    print("=" * 60)
    
    try:
        from scripts.run_crew_with_vetting import IntegratedSynthesisWorkflow
        
        # Initialize workflow
        workflow = IntegratedSynthesisWorkflow()
        
        if not workflow.components_ready:
            print("   ❌ Workflow components not ready")
            return False
        
        print("   🔄 Executing integrated synthesis workflow...")
        results = workflow.execute_complete_workflow(days_back=7)
        
        if results.get('success'):
            print("   ✅ Production workflow executed successfully")
            
            # Display summary
            vetting_summary = results.get('vetting_summary', {})
            print(f"   📊 Results Summary:")
            print(f"      Total hypotheses: {vetting_summary.get('total_hypotheses', 0)}")
            print(f"      Approved: {vetting_summary.get('approval_breakdown', {}).get('approved', 0)}")
            print(f"      Average score: {vetting_summary.get('average_score', 0):.1f}/100")
            
            return True
        else:
            error = results.get('error', 'Unknown error')
            print(f"   ❌ Production workflow failed: {error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Production workflow failed: {e}")
        return False

def main():
    """Run comprehensive CrewAI integration validation"""
    print("🎯 BULLETPROOF CREWAI INTEGRATION VALIDATION")
    print("NEVER SETTLE FOR LESS - GUARANTEED WORKING CREWAI")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run all tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("LLM Providers", test_llm_providers),
        ("CrewAI Basic Functionality", test_crewai_basic_functionality),
        ("Synthesis Agents", test_synthesis_agents),
        ("Vetting Agent", test_vetting_agent),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Final results
    print(f"\n{'='*80}")
    print(f"📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED - CREWAI FULLY INTEGRATED!")
        print(f"✅ System ready for production use")
        
        # Run production workflow
        print(f"\n🚀 ATTEMPTING PRODUCTION WORKFLOW...")
        if run_production_workflow():
            print(f"\n🎉 PRODUCTION WORKFLOW SUCCESSFUL!")
            print(f"✅ CrewAI integration is BULLETPROOF and PRODUCTION-READY")
        else:
            print(f"\n⚠️ Production workflow needs attention")
        
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED - FIXING REQUIRED")
        print(f"Fix the failing components before production use")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
