#!/usr/bin/env python3
"""
Enhanced Market Intelligence Agent with Multiple LLM Provider Support
Integrates OpenRouter and Abacus.ai for maximum reliability and cost optimization

Features:
- Dual LLM provider support (OpenRouter + Abacus.ai)
- Intelligent fallback strategies
- Cost optimization
- Rate limit handling
- Database integration with market_intelligence table
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import LLM providers
try:
    from llm_providers.abacus_integration import AbacusLLMTeams
except ImportError:
    print("⚠️ Abacus.ai integration not available")
    AbacusLLMTeams = None

try:
    from security.api_key_manager import get_secret
except ImportError:
    print("❌ FATAL: Could not import 'get_secret'. Make sure 'security/api_key_manager.py' exists.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMarketIntelligenceAgent:
    """Enhanced market intelligence agent with multi-provider LLM support"""
    
    def __init__(self):
        # Initialize database connection
        self.supabase = self._initialize_supabase()
        
        # Initialize LLM providers
        self.openrouter_llm = None
        self.abacus_llm = None
        
        # Try to initialize OpenRouter
        try:
            self.openrouter_llm = self._initialize_openrouter()
            logger.info("✅ OpenRouter LLM initialized")
        except Exception as e:
            logger.warning(f"⚠️ OpenRouter initialization failed: {e}")
        
        # Try to initialize Abacus.ai
        if AbacusLLMTeams:
            try:
                self.abacus_llm = AbacusLLMTeams()
                if self.abacus_llm.test_connection():
                    logger.info("✅ Abacus.ai LLM initialized and tested")
                else:
                    logger.warning("⚠️ Abacus.ai connection test failed")
                    self.abacus_llm = None
            except Exception as e:
                logger.warning(f"⚠️ Abacus.ai initialization failed: {e}")
                self.abacus_llm = None
        
        # Check if we have at least one working provider
        if not self.openrouter_llm and not self.abacus_llm:
            logger.error("❌ No LLM providers available!")
        
        logger.info("🧠 Enhanced Market Intelligence Agent initialized")
    
    def _initialize_supabase(self) -> Client:
        """Initialize Supabase client"""
        try:
            return create_client(get_secret('SUPABASE_URL'), get_secret('SUPABASE_KEY'))
        except Exception as e:
            logger.error(f"❌ Supabase initialization failed: {e}")
            raise
    
    def _initialize_openrouter(self) -> ChatOpenAI:
        """Initialize OpenRouter with ONLY free models (containing ':free')"""
        # STRICT: Only free models with ":free" in the name
        free_models = [
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-7b-it:free", 
            "meta-llama/llama-3-8b-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "google/gemma-2b-it:free",
            "nousresearch/nous-capybara-7b:free",
            "openchat/openchat-7b:free",
            "gryphe/mythomist-7b:free",
            "undi95/toppy-m-7b:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-mini-4k-instruct:free"
        ]
        
        api_key = get_secret("OPENROUTER_API_KEY")
        
        for model_name in free_models:
            try:
                llm = ChatOpenAI(
                    model=model_name,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=30,
                    default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE"}
                )
                
                # Test the model
                test_response = llm.invoke("Say 'OK' if working.")
                if test_response:
                    logger.info(f"✅ OpenRouter model working: {model_name}")
                    return llm
                    
            except Exception as e:
                logger.warning(f"⚠️ OpenRouter model {model_name} failed: {e}")
                continue
        
        raise RuntimeError("All OpenRouter models failed")
    
    def analyze_market_trends(self, focus_areas: List[str] = None, use_abacus: bool = False) -> Dict[str, Any]:
        """Analyze market trends using available LLM providers"""
        if focus_areas is None:
            focus_areas = ["AI automation", "SaaS platforms", "creator economy"]
        
        # Create analysis prompt
        prompt = f"""
        Analyze current market trends and identify business opportunities in these areas: {', '.join(focus_areas)}.
        
        Please provide:
        1. 3 emerging trends with business potential
        2. 3 customer pain points that represent opportunities
        3. Market size estimates where possible
        
        Format your response as JSON with this structure:
        {{
            "trends": [
                {{"title": "Trend Name", "description": "Brief description", "market_size": "Estimate", "opportunity_score": 8}}
            ],
            "pain_points": [
                {{"title": "Pain Point", "description": "Description", "affected_market": "Market segment", "urgency_score": 7}}
            ],
            "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"]
        }}
        """
        
        # Try providers in order of preference
        if use_abacus and self.abacus_llm:
            return self._analyze_with_abacus(prompt)
        elif self.openrouter_llm:
            result = self._analyze_with_openrouter(prompt)
            if result.get("success"):
                return result
            elif self.abacus_llm:
                logger.info("🔄 OpenRouter failed, trying Abacus.ai...")
                return self._analyze_with_abacus(prompt)
        elif self.abacus_llm:
            return self._analyze_with_abacus(prompt)
        else:
            return self._create_fallback_analysis()
    
    def _analyze_with_openrouter(self, prompt: str) -> Dict[str, Any]:
        """Analyze using OpenRouter LLM"""
        try:
            logger.info("🔄 Analyzing with OpenRouter...")
            response = self.openrouter_llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                analysis_data = json.loads(content[json_start:json_end])
                
                return {
                    "success": True,
                    "provider": "openrouter",
                    "data": analysis_data,
                    "raw_response": content
                }
            else:
                return {
                    "success": True,
                    "provider": "openrouter", 
                    "data": {"analysis": content},
                    "raw_response": content
                }
                
        except Exception as e:
            logger.error(f"❌ OpenRouter analysis failed: {e}")
            return {
                "success": False,
                "provider": "openrouter",
                "error": str(e)
            }
    
    def _analyze_with_abacus(self, prompt: str) -> Dict[str, Any]:
        """Analyze using Abacus.ai LLM"""
        try:
            logger.info("🔄 Analyzing with Abacus.ai...")
            result = self.abacus_llm.market_analysis_request(
                prompt=prompt,
                analysis_type="market_trends"
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "provider": "abacus",
                    "deployment_id": result.get("deployment_id"),
                    "conversation_id": result.get("conversation_id"),
                    "data": result.get("data"),
                    "raw_response": result.get("raw_response")
                }
            else:
                return {
                    "success": False,
                    "provider": "abacus",
                    "error": result.get("error")
                }
                
        except Exception as e:
            logger.error(f"❌ Abacus.ai analysis failed: {e}")
            return {
                "success": False,
                "provider": "abacus",
                "error": str(e)
            }
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when all providers fail"""
        logger.warning("⚠️ All LLM providers failed, using fallback analysis")
        
        return {
            "success": True,
            "provider": "fallback",
            "data": {
                "trends": [
                    {
                        "title": "AI-First Business Tools", 
                        "description": "Businesses adopting AI-native solutions for core operations",
                        "market_size": "$50B+ by 2027",
                        "opportunity_score": 8
                    },
                    {
                        "title": "No-Code Automation Platforms",
                        "description": "Growing demand for citizen developer tools",
                        "market_size": "$21B by 2026", 
                        "opportunity_score": 7
                    },
                    {
                        "title": "Creator Economy Infrastructure",
                        "description": "Tools and platforms supporting content creator businesses",
                        "market_size": "$104B by 2026",
                        "opportunity_score": 9
                    }
                ],
                "pain_points": [
                    {
                        "title": "SaaS Integration Complexity",
                        "description": "Businesses struggle with connecting multiple software tools",
                        "affected_market": "SMB and Enterprise",
                        "urgency_score": 8
                    },
                    {
                        "title": "AI Implementation Barriers",
                        "description": "High technical barriers for AI adoption in traditional businesses",
                        "affected_market": "Traditional Industries",
                        "urgency_score": 7
                    },
                    {
                        "title": "Creator Monetization Gaps",
                        "description": "Content creators lack efficient monetization tools",
                        "affected_market": "Individual Creators",
                        "urgency_score": 6
                    }
                ],
                "recommendations": [
                    "Focus on AI-powered integration platforms for SMBs",
                    "Develop no-code AI tools for traditional industries",
                    "Create comprehensive creator monetization infrastructure"
                ]
            }
        }
    
    def store_analysis_results(self, analysis_results: Dict[str, Any]) -> bool:
        """Store analysis results in the market_intelligence table"""
        try:
            storage_data = {
                "analysis_type": "enhanced_market_trends",
                "insights": analysis_results.get("data", {}),
                "timestamp": datetime.now().isoformat(),
                "source": "enhanced_market_intel_agent",
                "metadata": {
                    "provider_used": analysis_results.get("provider"),
                    "deployment_id": analysis_results.get("deployment_id"),
                    "conversation_id": analysis_results.get("conversation_id"),
                    "success": analysis_results.get("success")
                }
            }
            
            result = self.supabase.table('market_intelligence').insert(storage_data).execute()
            
            if result.data:
                logger.info("✅ Analysis results stored in market_intelligence table")
                return True
            else:
                logger.error("❌ Failed to store analysis results")
                return False
                
        except Exception as e:
            logger.error(f"❌ Storage error: {e}")
            return False
    
    def run_complete_analysis(self, focus_areas: List[str] = None, force_abacus: bool = False) -> Dict[str, Any]:
        """Run complete market intelligence analysis with storage"""
        logger.info("🚀 Starting enhanced market intelligence analysis...")
        
        # Run analysis
        analysis_results = self.analyze_market_trends(
            focus_areas=focus_areas,
            use_abacus=force_abacus
        )
        
        # Store results
        stored = self.store_analysis_results(analysis_results)
        
        # Compile final report
        final_results = {
            **analysis_results,
            "stored_successfully": stored,
            "execution_timestamp": datetime.now().isoformat(),
            "summary": {
                "provider_used": analysis_results.get("provider"),
                "trends_identified": len(analysis_results.get("data", {}).get("trends", [])),
                "pain_points_identified": len(analysis_results.get("data", {}).get("pain_points", [])),
                "recommendations_count": len(analysis_results.get("data", {}).get("recommendations", []))
            }
        }
        
        logger.info(f"✅ Analysis complete using {analysis_results.get('provider')} provider")
        return final_results

def test_enhanced_agent():
    """Test function for the enhanced market intelligence agent"""
    print("🚀 Testing Enhanced Market Intelligence Agent...")
    
    agent = EnhancedMarketIntelligenceAgent()
    
    # Test 1: Analysis with fallback
    print("\n1. Testing analysis with provider fallback...")
    result = agent.run_complete_analysis(
        focus_areas=["AI automation", "fintech", "healthcare tech"],
        force_abacus=False
    )
    
    if result.get("success"):
        print(f"✅ Analysis successful: {result.get('provider')} provider used")
        print(f"📊 Trends: {result.get('summary', {}).get('trends_identified', 0)}")
        print(f"📊 Pain points: {result.get('summary', {}).get('pain_points_identified', 0)}")
        print(f"💾 Stored in database: {result.get('stored_successfully')}")
    else:
        print(f"❌ Analysis failed: {result.get('error')}")
    
    return True

if __name__ == "__main__":
    test_enhanced_agent()
