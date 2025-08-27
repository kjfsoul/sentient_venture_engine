#!/usr/bin/env python3
"""
Ultimate Multi-Provider Market Intelligence Agent
Uses ALL available LLM providers with intelligent routing and fallback strategies

Providers:
1. OpenRouter (premium models with free tier)
2. Abacus.ai LLM Teams (enterprise-grade)
3. RouteLL (intelligent model routing)
4. Together.ai (open-source models)

Features:
- Intelligent provider selection based on task complexity
- Automatic fallback across all providers
- Cost optimization and performance monitoring
- Database integration with market_intelligence table
- Enhanced error handling and logging
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import all LLM providers
try:
    from llm_providers.abacus_integration import AbacusLLMTeams
except ImportError:
    print("⚠️ Abacus.ai integration not available")
    AbacusLLMTeams = None

try:
    from llm_providers.routell_integration import RouteLLProvider
except ImportError:
    print("⚠️ RouteLL integration not available")
    RouteLLProvider = None

try:
    from llm_providers.together_integration import TogetherAIProvider
except ImportError:
    print("⚠️ Together.ai integration not available")
    TogetherAIProvider = None

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

class TaskComplexity(Enum):
    """Task complexity levels for intelligent provider selection"""
    SIMPLE = "simple"      # Basic queries, simple analysis
    MEDIUM = "medium"      # Market analysis, trend identification
    COMPLEX = "complex"    # Deep synthesis, business model design
    CRITICAL = "critical"  # Mission-critical analysis requiring best models

class ProviderPriority(Enum):
    """Provider priority levels for different scenarios"""
    COST_OPTIMIZED = "cost"      # Prefer cheaper providers
    PERFORMANCE = "performance"   # Prefer highest quality models
    SPEED = "speed"              # Prefer fastest response
    BALANCED = "balanced"        # Balance cost, performance, speed

class UltimateMarketIntelligenceAgent:
    """Ultimate market intelligence agent with all LLM providers"""
    
    def __init__(self):
        # Initialize database connection
        self.supabase = self._initialize_supabase()
        
        # Initialize all available LLM providers
        self.providers = {}
        self.provider_status = {}
        
        # Initialize OpenRouter
        try:
            self.providers['openrouter'] = self._initialize_openrouter()
            self.provider_status['openrouter'] = 'available'
            logger.info("✅ OpenRouter LLM initialized")
        except Exception as e:
            logger.warning(f"⚠️ OpenRouter initialization failed: {e}")
            self.provider_status['openrouter'] = 'failed'
        
        # Initialize Abacus.ai
        if AbacusLLMTeams:
            try:
                abacus_provider = AbacusLLMTeams()
                if abacus_provider.test_connection():
                    self.providers['abacus'] = abacus_provider
                    self.provider_status['abacus'] = 'available'
                    logger.info("✅ Abacus.ai LLM initialized and tested")
                else:
                    self.provider_status['abacus'] = 'connection_failed'
                    logger.warning("⚠️ Abacus.ai connection test failed")
            except Exception as e:
                logger.warning(f"⚠️ Abacus.ai initialization failed: {e}")
                self.provider_status['abacus'] = 'failed'
        else:
            self.provider_status['abacus'] = 'not_available'
        
        # Initialize RouteLL
        if RouteLLProvider:
            try:
                routell_provider = RouteLLProvider()
                if routell_provider.test_connection():
                    self.providers['routell'] = routell_provider
                    self.provider_status['routell'] = 'available'
                    logger.info("✅ RouteLL LLM initialized and tested")
                else:
                    self.provider_status['routell'] = 'connection_failed'
                    logger.warning("⚠️ RouteLL connection test failed")
            except Exception as e:
                logger.warning(f"⚠️ RouteLL initialization failed: {e}")
                self.provider_status['routell'] = 'failed'
        else:
            self.provider_status['routell'] = 'not_available'
        
        # Initialize Together.ai
        if TogetherAIProvider:
            try:
                together_provider = TogetherAIProvider()
                if together_provider.test_connection():
                    self.providers['together'] = together_provider
                    self.provider_status['together'] = 'available'
                    logger.info("✅ Together.ai LLM initialized and tested")
                else:
                    self.provider_status['together'] = 'connection_failed'
                    logger.warning("⚠️ Together.ai connection test failed")
            except Exception as e:
                logger.warning(f"⚠️ Together.ai initialization failed: {e}")
                self.provider_status['together'] = 'failed'
        else:
            self.provider_status['together'] = 'not_available'
        
        # Check if we have at least one working provider
        available_providers = [name for name, status in self.provider_status.items() 
                             if status == 'available']
        
        if not available_providers:
            logger.error("❌ No LLM providers available!")
        else:
            logger.info(f"🧠 Ultimate Market Intelligence Agent initialized with {len(available_providers)} providers: {available_providers}")
        
        # Provider performance tracking
        self.provider_performance = {name: {'success_rate': 1.0, 'avg_response_time': 0.0, 'total_requests': 0}
                                   for name in self.provider_status.keys()}
    
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
    
    def get_provider_priority(self, 
                             complexity: TaskComplexity, 
                             priority: ProviderPriority) -> List[str]:
        """Get ordered list of providers based on task complexity and priority"""
        
        # Base provider capabilities
        provider_rankings = {
            TaskComplexity.SIMPLE: {
                ProviderPriority.COST_OPTIMIZED: ['together', 'openrouter', 'routell', 'abacus'],
                ProviderPriority.SPEED: ['together', 'openrouter', 'routell', 'abacus'],
                ProviderPriority.PERFORMANCE: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.BALANCED: ['openrouter', 'together', 'routell', 'abacus']
            },
            TaskComplexity.MEDIUM: {
                ProviderPriority.COST_OPTIMIZED: ['openrouter', 'together', 'routell', 'abacus'],
                ProviderPriority.SPEED: ['openrouter', 'together', 'routell', 'abacus'],
                ProviderPriority.PERFORMANCE: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.BALANCED: ['routell', 'openrouter', 'abacus', 'together']
            },
            TaskComplexity.COMPLEX: {
                ProviderPriority.COST_OPTIMIZED: ['routell', 'openrouter', 'abacus', 'together'],
                ProviderPriority.SPEED: ['routell', 'abacus', 'openrouter', 'together'],
                ProviderPriority.PERFORMANCE: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.BALANCED: ['abacus', 'routell', 'openrouter', 'together']
            },
            TaskComplexity.CRITICAL: {
                ProviderPriority.COST_OPTIMIZED: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.SPEED: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.PERFORMANCE: ['abacus', 'routell', 'openrouter', 'together'],
                ProviderPriority.BALANCED: ['abacus', 'routell', 'openrouter', 'together']
            }
        }
        
        # Get the priority order and filter by available providers
        priority_order = provider_rankings[complexity][priority]
        available_providers = [name for name in priority_order 
                             if self.provider_status.get(name) == 'available']
        
        # Add performance-based reordering
        if len(available_providers) > 1:
            available_providers.sort(key=lambda x: (
                -self.provider_performance[x]['success_rate'],  # Higher success rate first
                self.provider_performance[x]['avg_response_time']  # Lower response time first
            ))
        
        return available_providers
    
    def analyze_with_fallback(self, 
                             prompt: str,
                             complexity: TaskComplexity = TaskComplexity.MEDIUM,
                             priority: ProviderPriority = ProviderPriority.BALANCED,
                             focus_areas: List[str] = None) -> Dict[str, Any]:
        """Analyze using intelligent provider selection with fallback"""
        
        if focus_areas is None:
            focus_areas = ["market trends", "business opportunities"]
        
        # Get provider priority order
        provider_order = self.get_provider_priority(complexity, priority)
        
        if not provider_order:
            return self._create_fallback_analysis("No providers available")
        
        logger.info(f"🎯 Task: {complexity.value}, Priority: {priority.value}")
        logger.info(f"🔄 Provider order: {provider_order}")
        
        # Try providers in order
        for provider_name in provider_order:
            try:
                start_time = datetime.now()
                logger.info(f"🔄 Trying provider: {provider_name}")
                
                result = self._analyze_with_provider(provider_name, prompt, focus_areas)
                
                if result.get('success'):
                    # Update performance metrics
                    response_time = (datetime.now() - start_time).total_seconds()
                    self._update_performance_metrics(provider_name, True, response_time)
                    
                    logger.info(f"✅ Analysis successful with {provider_name}")
                    return result
                else:
                    # Update performance metrics for failure
                    self._update_performance_metrics(provider_name, False, 0)
                    logger.warning(f"⚠️ {provider_name} failed: {result.get('error')}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ {provider_name} exception: {e}")
                self._update_performance_metrics(provider_name, False, 0)
                continue
        
        # All providers failed
        logger.error("❌ All providers failed, using fallback analysis")
        return self._create_fallback_analysis("All providers failed")
    
    def _analyze_with_provider(self, provider_name: str, prompt: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze using specific provider"""
        provider = self.providers.get(provider_name)
        
        if not provider:
            return {"success": False, "error": f"Provider {provider_name} not available"}
        
        if provider_name == 'openrouter':
            return self._analyze_with_openrouter(prompt, focus_areas)
        elif provider_name == 'abacus':
            return provider.market_analysis_request(prompt)
        elif provider_name == 'routell':
            return provider.market_analysis_request(prompt)
        elif provider_name == 'together':
            return provider.market_analysis_request(prompt)
        else:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}
    
    def _analyze_with_openrouter(self, prompt: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze using OpenRouter LLM"""
        try:
            openrouter_llm = self.providers['openrouter']
            response = openrouter_llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                try:
                    analysis_data = json.loads(content[json_start:json_end])
                    return {
                        "success": True,
                        "provider": "openrouter",
                        "data": analysis_data,
                        "raw_response": content
                    }
                except json.JSONDecodeError:
                    pass
            
            return {
                "success": True,
                "provider": "openrouter", 
                "data": {"analysis": content},
                "raw_response": content
            }
                
        except Exception as e:
            return {"success": False, "provider": "openrouter", "error": str(e)}
    
    def _create_fallback_analysis(self, reason: str) -> Dict[str, Any]:
        """Create minimal fallback analysis when all providers fail"""
        logger.warning(f"Using fallback analysis - {reason}")
        return {
            "success": True,
            "provider": "fallback",
            "data": {
                "trends": [{"title": "Analysis unavailable", "summary": "All providers failed", "url": "System"}],
                "pain_points": [{"title": "Analysis unavailable", "summary": "All providers failed", "url": "System"}],
                "note": f"Fallback mode - {reason}"
            },
            "raw_response": f"Fallback analysis due to: {reason}"
        }
    
    def _update_performance_metrics(self, provider_name: str, success: bool, response_time: float):
        """Update performance metrics for provider"""
        metrics = self.provider_performance[provider_name]
        metrics['total_requests'] += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        metrics['success_rate'] = (1 - alpha) * metrics['success_rate'] + alpha * (1.0 if success else 0.0)
        
        # Update average response time
        if success and response_time > 0:
            if metrics['avg_response_time'] == 0:
                metrics['avg_response_time'] = response_time
            else:
                metrics['avg_response_time'] = (1 - alpha) * metrics['avg_response_time'] + alpha * response_time
    
    def store_analysis_results(self, analysis_results: Dict[str, Any]) -> bool:
        """Store analysis results in the market_intelligence table"""
        try:
            storage_data = {
                "analysis_type": "ultimate_market_intelligence",
                "insights": analysis_results.get("data", {}),
                "timestamp": datetime.now().isoformat(),
                "source": "ultimate_market_intel_agent",
                "metadata": {
                    "provider_used": analysis_results.get("provider"),
                    "success": analysis_results.get("success"),
                    "complexity": "medium",  # Could be parameterized
                    "priority": "balanced"
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
    
    def get_provider_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of all providers"""
        return {
            "total_providers": len(self.provider_status),
            "available_providers": [name for name, status in self.provider_status.items() 
                                  if status == 'available'],
            "provider_status": self.provider_status,
            "performance_metrics": self.provider_performance,
            "recommendations": self._get_provider_recommendations()
        }
    
    def _get_provider_recommendations(self) -> List[str]:
        """Get recommendations for improving provider setup"""
        recommendations = []
        
        available_count = len([s for s in self.provider_status.values() if s == 'available'])
        
        if available_count == 0:
            recommendations.append("CRITICAL: No LLM providers available. Check all API keys.")
        elif available_count == 1:
            recommendations.append("WARNING: Only one provider available. Add more for redundancy.")
        elif available_count == 2:
            recommendations.append("GOOD: Two providers available. Consider adding more for optimal redundancy.")
        else:
            recommendations.append("EXCELLENT: Multiple providers available for optimal redundancy.")
        
        # Check specific providers
        if self.provider_status.get('abacus') == 'connection_failed':
            recommendations.append("Add Abacus.ai credentials to .env for enterprise-grade models.")
        
        if self.provider_status.get('routell') == 'connection_failed':
            recommendations.append("Verify RouteLL API key for intelligent model routing.")
        
        if self.provider_status.get('together') == 'connection_failed':
            recommendations.append("Check Together.ai API key for cost-effective open-source models.")
        
        return recommendations
    
    def run_comprehensive_analysis(self, 
                                  focus_areas: List[str] = None,
                                  complexity: TaskComplexity = TaskComplexity.MEDIUM,
                                  priority: ProviderPriority = ProviderPriority.BALANCED) -> Dict[str, Any]:
        """Run comprehensive market intelligence analysis with storage"""
        
        if focus_areas is None:
            focus_areas = ["AI automation", "market intelligence", "business opportunities"]
        
        # Create comprehensive prompt
        prompt = f"""
        Analyze current market trends and identify business opportunities in these areas: {', '.join(focus_areas)}.
        
        Please provide a comprehensive analysis including:
        1. 3 emerging trends with business potential
        2. 3 customer pain points that represent opportunities
        3. Market size estimates where possible
        4. Competitive landscape insights
        5. Implementation recommendations
        
        Format your response as JSON with this structure:
        {{
            "trends": [
                {{"title": "Trend Name", "description": "Brief description", "market_size": "Estimate", "opportunity_score": 8}}
            ],
            "pain_points": [
                {{"title": "Pain Point", "description": "Description", "affected_market": "Market segment", "urgency_score": 7}}
            ],
            "competitive_insights": [
                {{"area": "Market Area", "key_players": ["Player1", "Player2"], "gaps": "Market gaps"}}
            ],
            "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"]
        }}
        """
        
        logger.info("🚀 Starting comprehensive market intelligence analysis...")
        
        # Perform analysis with intelligent provider selection
        analysis_result = self.analyze_with_fallback(prompt, complexity, priority, focus_areas)
        
        # Store results in database
        stored = self.store_analysis_results(analysis_result)
        
        # Add metadata to result
        analysis_result.update({
            "stored_in_db": stored,
            "focus_areas": focus_areas,
            "analysis_timestamp": datetime.now().isoformat(),
            "provider_status": self.get_provider_status_report()
        })
        
        return analysis_result

# Test the ultimate agent
if __name__ == "__main__":
    print("🚀 Testing Ultimate Multi-Provider Market Intelligence Agent")
    print("=" * 70)
    
    agent = UltimateMarketIntelligenceAgent()
    
    # Get status report
    status = agent.get_provider_status_report()
    print(f"\n📊 Provider Status:")
    print(f"  Available: {status['available_providers']}")
    print(f"  Total: {status['total_providers']}")
    print(f"\n💡 Recommendations:")
    for rec in status['recommendations']:
        print(f"  • {rec}")
    
    # Run comprehensive analysis
    print(f"\n🔍 Running comprehensive analysis...")
    result = agent.run_comprehensive_analysis(
        focus_areas=["AI automation", "small business tools"],
        complexity=TaskComplexity.MEDIUM,
        priority=ProviderPriority.BALANCED
    )
    
    print(f"\n✅ Analysis Results:")
    print(f"  Success: {result.get('success')}")
    print(f"  Provider: {result.get('provider')}")
    print(f"  Stored: {result.get('stored_in_db')}")
    print(f"  Data keys: {list(result.get('data', {}).keys())}")
