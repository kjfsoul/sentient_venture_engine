#!/usr/bin/env python3
"""
Enhanced Market Intelligence Agent with Multi-Provider LLM Support

Features:
- Support for Gemini Advanced, ChatGPT Plus, and other providers
- Intelligent routing based on task complexity and cost optimization
- Resilient integration with fallback mechanisms
- Unified agent architecture for consistent output
- Rate limiting and failure handling
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from functools import wraps

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# Import your secrets manager
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

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'default': {'requests_per_minute': 10, 'burst_limit': 5},
    'openai': {'requests_per_minute': 10, 'burst_limit': 5},
    'gemini': {'requests_per_minute': 5, 'burst_limit': 3},
    'openrouter': {'requests_per_minute': 15, 'burst_limit': 8}
}

# Global rate limiting tracker
rate_limit_tracker = {}

def rate_limit(provider='default'):
    """Rate limiting decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RATE_LIMIT_CONFIG.get(provider, RATE_LIMIT_CONFIG['default'])
            requests_per_minute = config['requests_per_minute']
            
            # Initialize tracker for this provider
            if provider not in rate_limit_tracker:
                rate_limit_tracker[provider] = []
            
            now = time.time()
            # Remove requests older than 1 minute
            rate_limit_tracker[provider] = [
                req_time for req_time in rate_limit_tracker[provider] 
                if now - req_time < 60
            ]
            
            # Check if we're at the rate limit
            if len(rate_limit_tracker[provider]) >= requests_per_minute:
                # Calculate wait time
                oldest_request = min(rate_limit_tracker[provider])
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    logger.warning(f"⏳ Rate limit reached for {provider}, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
            
            # Record this request
            rate_limit_tracker[provider].append(now)
            
            # Call the function
            return func(*args, **kwargs)
        return wrapper
    return decorator

def exponential_backoff(max_retries=3, base_delay=1.0):
    """Exponential backoff decorator for handling API failures"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Last attempt, re-raise the exception
                        raise e
                    
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    if 'rate' in error_str or '429' in error_str:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + (0.1 * (attempt + 1))
                        jitter = 0.1 * delay * (2 * (hash(str(attempt)) % 1000) / 1000 - 1)
                        total_delay = max(0, delay + jitter)
                        
                        logger.warning(f"⚠️ Rate limit hit on attempt {attempt + 1}, backing off for {total_delay:.2f} seconds")
                        time.sleep(total_delay)
                    else:
                        # Non-rate limit error, re-raise immediately
                        raise e
            return None
        return wrapper
    return decorator

class EnhancedMarketIntelAgent:
    """Enhanced market intelligence agent with multi-provider LLM support"""
    
    def __init__(self):
        # Initialize all provider keys
        self.openrouter_key = get_secret('OPENROUTER_API_KEY')
        self.openai_key = get_secret('OPENAI_API_KEY')
        self.gemini_key = get_secret('GEMINI_API_KEY')
        
        # Supabase setup
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase: {e}")
        
        # Multi-provider LLM models with cost optimization
        self.providers = {
            'openrouter': {
                'models': [
                    "anthropic/claude-3-5-sonnet",  # High performance
                    "openai/gpt-4o",                # High performance
                    "google/gemini-pro",            # High performance
                    "meta-llama/llama-3.1-70b-instruct",  # Mid performance
                    "anthropic/claude-3-haiku:free",      # Free tier
                    "openai/gpt-4o-mini:free",            # Free tier
                    "google/gemini-flash-1.5:free"        # Free tier
                ],
                'base_url': "https://openrouter.ai/api/v1",
                'api_key': self.openrouter_key
            },
            'openai': {
                'models': [
                    "gpt-4-turbo",     # High performance
                    "gpt-4",           # High performance
                    "gpt-3.5-turbo"    # Mid performance
                ],
                'base_url': "https://api.openai.com/v1",
                'api_key': self.openai_key
            },
            'gemini': {
                'models': [
                    "gemini-1.5-pro",  # High performance (Gemini Advanced)
                    "gemini-1.5-flash", # Mid performance
                    "gemini-pro"       # Mid performance
                ],
                'base_url': "https://generativelanguage.googleapis.com/v1beta",
                'api_key': self.gemini_key
            }
        }
        
        # Task complexity mapping for intelligent routing
        self.task_complexity = {
            'simple': ['basic_analysis', 'data_extraction', 'simple_classification'],
            'moderate': ['market_analysis', 'competitive_analysis', 'trend_identification'],
            'complex': ['strategic_planning', 'business_model_design', 'comprehensive_synthesis'],
            'critical': ['financial_modeling', 'risk_assessment', 'high_stakes_decisions']
        }

    @rate_limit('openai')
    @exponential_backoff(max_retries=3)
    def initialize_llm(self, task_type: str = 'moderate', provider_preference: str = 'openrouter') -> ChatOpenAI:
        """Initialize LLM with intelligent routing based on task complexity and cost optimization"""
        
        # Determine model tier based on task complexity
        if task_type in self.task_complexity['simple']:
            model_tier = 'free'
            max_tokens = 800
            temperature = 0.3
        elif task_type in self.task_complexity['moderate']:
            model_tier = 'mid'
            max_tokens = 1200
            temperature = 0.5
        elif task_type in self.task_complexity['complex']:
            model_tier = 'high'
            max_tokens = 2000
            temperature = 0.7
        else:  # critical
            model_tier = 'high'
            max_tokens = 3000
            temperature = 0.2  # More deterministic for critical tasks
        
        # Provider priority order
        provider_order = [provider_preference, 'openrouter', 'openai', 'gemini']
        
        # Try each provider in order
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
                
            provider = self.providers[provider_name]
            models = provider['models']
            
            # Filter models based on tier
            if model_tier == 'free':
                filtered_models = [m for m in models if ':free' in m or 'flash' in m.lower()]
            elif model_tier == 'mid':
                filtered_models = [m for m in models if 'gpt-4o-mini' in m or 'haiku' in m or 'flash' in m.lower() or 'pro' in m]
            else:  # high
                filtered_models = [m for m in models if 'sonnet' in m or 'gpt-4' in m or 'pro' in m]
            
            # Try each model
            for model in filtered_models:
                try:
                    logger.info(f"🔄 Trying LLM: {provider_name}/{model}")
                    
                    # Special handling for Gemini API
                    if provider_name == 'gemini':
                        llm = ChatOpenAI(
                            model=model,
                            openai_api_key=provider['api_key'],
                            openai_api_base=f"{provider['base_url']}/models/{model}:generateContent",
                            temperature=temperature,
                            max_tokens=max_tokens,
                            default_headers={
                                "Authorization": f"Bearer {provider['api_key']}",
                                "Content-Type": "application/json"
                            }
                        )
                    else:
                        llm = ChatOpenAI(
                            model=model,
                            api_key=provider['api_key'],
                            base_url=provider['base_url'],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            default_headers={
                                "HTTP-Referer": "https://sve.ai", 
                                "X-Title": "SVE-Enhanced"
                            }
                        )
                    
                    # Smoke test
                    test_response = llm.invoke("Say 'OK' if you're working.")
                    logger.info(f"✅ Successfully initialized LLM: {provider_name}/{model}")
                    return llm
                    
                except Exception as e:
                    logger.warning(f"⚠️ Model {provider_name}/{model} failed: {str(e)[:100]}...")
                    # Re-raise rate limit errors for backoff
                    if "rate" in str(e).lower() or "429" in str(e):
                        raise e
                    continue
        
        # If all providers fail, raise an error
        raise RuntimeError("❌ FATAL: All LLM providers and models failed initialization")

    def run_enhanced_market_analysis(self, task_type: str = 'moderate') -> Dict[str, Any]:
        """Run enhanced market analysis with multi-provider LLM support"""
        logger.info("🚀 Starting Enhanced Market Intelligence Analysis")
        
        try:
            # Initialize LLM with appropriate provider
            llm = self.initialize_llm(task_type)
            
            # Initialize search tool
            search_tool = None
            disable_search = os.getenv("DISABLE_SEARCH", "false").lower() == "true"
            if not disable_search:
                try:
                    search_tool = DuckDuckGoSearchRun()
                except Exception as e:
                    logger.warning(f"⚠️ Search tool initialization failed: {e}")
            
            # Prepare analysis task
            if search_tool:
                task = """
                Find emerging business opportunities in SaaS/AI/creator economy.
                
                Instructions:
                1. Make ONE search for "emerging trends SaaS AI creator economy 2025" 
                2. Make ONE search for "customer pain points SaaS AI problems"
                3. Return ONLY this JSON format:
                
                {
                  "trends": [
                    {"title": "Brief Title", "summary": "One sentence.", "url": "source_url"}
                  ],
                  "pain_points": [
                    {"title": "Brief Title", "summary": "One sentence.", "url": "source_url"}
                  ]
                }
                """
            else:
                task = """
                Based on your knowledge, identify business opportunities:
                1. List 3 trends in SaaS/AI/creator economy
                2. List 3 customer pain points
                3. Return ONLY this JSON format:
                
                {
                  "trends": [
                    {"title": "Brief Title", "summary": "One sentence.", "url": "AI Knowledge Base"}
                  ],
                  "pain_points": [
                    {"title": "Brief Title", "summary": "One sentence.", "url": "AI Knowledge Base"}
                  ]
                }
                """
            
            # Execute analysis
            if search_tool:
                # Use agent with tools
                from langchain.agents import initialize_agent, AgentType
                import warnings
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                
                agent_executor = initialize_agent(
                    tools=[search_tool],
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=2,
                    max_execution_time=30,
                    early_stopping_method="generate"
                )
                
                result = agent_executor.invoke({"input": task})
                output = result.get("output", "")
            else:
                # Use LLM directly
                response = llm.invoke(task)
                output = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON output
            data = self._parse_json_output(output)
            
            # Store results
            stored = self._store_market_intelligence(data)
            
            # Return results
            final_results = {
                'success': True,
                'trends': data.get('trends', []),
                'pain_points': data.get('pain_points', []),
                'stored_successfully': stored,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Enhanced Market Intelligence Analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced Market Intelligence Analysis failed: {e}")
            return {
                "error": str(e), 
                "success": False,
                "trends": [],
                "pain_points": [],
                "stored_successfully": False,
                "execution_timestamp": datetime.now().isoformat()
            }

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """Parse JSON output from LLM with fallback mechanisms"""
        try:
            # Try to find JSON in the output
            json_start = output.find('{')
            json_end = output.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = output[json_start:json_end+1]
                return json.loads(json_str)
            else:
                logger.warning("No JSON structure found in output")
                return {"trends": [], "pain_points": []}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {"trends": [], "pain_points": []}

    def _store_market_intelligence(self, data: Dict[str, Any]) -> bool:
        """Store market intelligence in Supabase"""
        if not self.supabase:
            logger.warning("Supabase not available - data not stored")
            return False
        
        try:
            stored_count = 0
            
            # Store trends
            for trend in data.get("trends", []):
                if isinstance(trend, dict) and trend.get('title'):
                    db_payload = {
                        'type': 'trend',
                        'source_url': trend.get('url', ''),
                        'processed_insights_path': f"{trend.get('title')} - {trend.get('summary', '')}"
                    }
                    self.supabase.table('data_sources').insert(db_payload).execute()
                    stored_count += 1
            
            # Store pain points
            for pain in data.get("pain_points", []):
                if isinstance(pain, dict) and pain.get('title'):
                    db_payload = {
                        'type': 'pain_point',
                        'source_url': pain.get('url', ''),
                        'processed_insights_path': f"{pain.get('title')} - {pain.get('summary', '')}"
                    }
                    self.supabase.table('data_sources').insert(db_payload).execute()
                    stored_count += 1
            
            logger.info(f"✅ Stored {stored_count} market intelligence items")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store market intelligence: {e}")
            return False

def main():
    """Main execution function"""
    print("📈 Starting Enhanced Market Intelligence Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = EnhancedMarketIntelAgent()
    
    # Run analysis
    results = agent.run_enhanced_market_analysis()
    
    # Display results
    if results.get('success'):
        print("\n✅ ENHANCED MARKET INTELLIGENCE ANALYSIS COMPLETE")
        print(f"📊 Trends identified: {len(results['trends'])}")
        print(f"🔍 Pain points identified: {len(results['pain_points'])}")
        print(f"💾 Data stored: {results['stored_successfully']}")
        
        if results['trends']:
            print(f"\n📈 TOP TRENDS:")
            for i, trend in enumerate(results['trends'][:3], 1):
                print(f"   {i}. {trend.get('title', 'Unknown')}")
                print(f"      {trend.get('summary', 'No summary')}")
        
        if results['pain_points']:
            print(f"\n痛点识别:")
            for i, pain in enumerate(results['pain_points'][:3], 1):
                print(f"   {i}. {pain.get('title', 'Unknown')}")
                print(f"      {pain.get('summary', 'No summary')}")
        
        print(f"\n🎉 Enhanced market intelligence gathering complete!")
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
