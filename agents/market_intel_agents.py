# sentient_venture_engine/agents/market_intel_agents.py

import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# Import your secrets manager
try:
    from security.api_key_manager import get_secret
except ImportError:
    print("❌ FATAL: Could not import 'get_secret'. Make sure 'security/api_key_manager.py' exists.")
    sys.exit(1)

# --- Global Initializations ---
def initialize_llm():
    """Initializes the Language Model with explicit routing for OpenRouter using free models with fallback strategy."""
    
    # Comprehensive list of free OpenRouter models in order of preference
    free_models = [
        "microsoft/phi-3-mini-128k-instruct:free",
        "google/gemma-7b-it:free", 
        "meta-llama/llama-3-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "google/gemma-2b-it:free",
        "nousresearch/nous-capybara-7b:free",
        "openchat/openchat-7b:free",
        "gryphe/mythomist-7b:free",
        "undi95/toppy-m-7b:free",
        "openrouter/auto",  # Auto-select available free model
        "meta-llama/llama-3.1-8b-instruct:free",
        "microsoft/phi-3-mini-4k-instruct:free"
    ]
    
    # Environment variable override for primary model
    primary_model = os.getenv("LLM_MODEL", free_models[0])
    
    # Create try order with primary model first, then remaining free models
    try_order = [primary_model] + [m for m in free_models if m != primary_model]
    
    # Conservative defaults for free models (increased to reduce truncation)
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    api_key = get_secret("OPENROUTER_API_KEY")
    
    last_error = None
    for model_name in try_order:
        try:
            print(f"🔄 Trying free model: {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE"}
            )
            
            # Smoke test to verify the model works
            test_response = llm.invoke("Say 'OK' if you're working.")
            print(f"✅ Successfully initialized free model: {model_name}")
            return llm
            
        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {str(e)[:100]}...")
            last_error = e
            continue
    
    # If all models fail, raise the last error
    error_msg = f"❌ FATAL: All free models failed. Last error: {last_error}"
    print(error_msg)
    raise RuntimeError(error_msg)

def initialize_supabase_client():
    """Initializes and returns the Supabase client."""
    try:
        return create_client(get_secret('SUPABASE_URL'), get_secret('SUPABASE_KEY'))
    except Exception as e:
        print(f"❌ FATAL: Failed to initialize Supabase. Check credentials. Error: {e}")
        sys.exit(1)

def initialize_search_tool():
    """Initialize search tool with retry logic for rate limiting."""
    
    # Check if search is disabled via environment variable
    disable_search = os.getenv("DISABLE_SEARCH", "false").lower() == "true"
    if disable_search:
        print("🚫 Search disabled via DISABLE_SEARCH environment variable")
        return None
        
    try:
        # Add a small delay to help with rate limiting
        import time
        time.sleep(2)  # 2-second delay before initializing search
        return DuckDuckGoSearchRun()
    except Exception as e:
        print(f"⚠️ DuckDuckGo search initialization failed: {e}")
        print("Continuing without search tool - agent will work with knowledge only")
        return None

llm = initialize_llm()
supabase = initialize_supabase_client()
search_tool = initialize_search_tool()

# --- Agent Logic ---
def run_market_analysis():
    """
    Runs a simple, powerful LangChain agent to find market trends and pain points.
    """
    print("🚀 Kicking off Market Intelligence Agent...")

    # Check for test mode to avoid rate limiting
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    if test_mode:
        print("🧪 Running in TEST MODE - using knowledge only, no search")
        # In test mode, skip agent entirely and generate sample data
        data = {
            "trends": [
                {"title": "AI-Powered SaaS Analytics", "summary": "SaaS companies adopting AI for predictive customer analytics.", "url": "Test Knowledge Base"},
                {"title": "No-Code Movement Expansion", "summary": "Growing adoption of no-code platforms for business automation.", "url": "Test Knowledge Base"},
                {"title": "Creator Economy Tools", "summary": "New platforms emerging for creator monetization and audience management.", "url": "Test Knowledge Base"}
            ],
            "pain_points": [
                {"title": "SaaS Integration Complexity", "summary": "Businesses struggle with connecting multiple SaaS tools effectively.", "url": "Test Knowledge Base"},
                {"title": "Creator Payment Delays", "summary": "Content creators face delayed payments from platform monetization.", "url": "Test Knowledge Base"},
                {"title": "AI Model Costs", "summary": "Small businesses find AI API costs prohibitive for regular use.", "url": "Test Knowledge Base"}
            ]
        }
        print("📊 Generated test data for development purposes")
    else:
        # Regular agent execution
        # Use initialize_agent with deprecation warning suppression
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Prepare tools list - handle case where search tool is unavailable
        tools = [search_tool] if search_tool else []
        
        if not tools:
            print("⚠️ No search tools available due to rate limiting. Agent will work with internal knowledge only.")
        
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,  # Reduced to minimize search requests
            max_execution_time=30,  # Reduced timeout
            early_stopping_method="generate"  # Stop when final answer is generated
        )

        # Adjust task based on available tools
        if search_tool:
            task = """
            IMPORTANT: You must provide a complete JSON response. Minimize search requests to avoid rate limits.
            
            Task: Find emerging business opportunities in SaaS/AI/creator economy.
            
            Instructions:
            1. Make ONE broad search for "emerging trends SaaS AI creator economy 2024" 
            2. Make ONE search for "customer pain points SaaS AI problems"
            3. Based on results, identify 3 trends and 3 pain points
            4. Return ONLY this JSON format (no explanations):
            
            {
              "trends": [
                {"title": "Brief Title", "summary": "One sentence.", "url": "source_url"}
              ],
              "pain_points": [
                {"title": "Brief Title", "summary": "One sentence.", "url": "source_url"}
              ]
            }
            
            CRITICAL: Use maximum 2 search actions total. Be efficient.
            """
        else:
            task = """
            IMPORTANT: You must provide a complete JSON response. Do not explain your process.
            
            Based on your knowledge, identify business opportunities:
            1. List 3 trends in SaaS/AI/creator economy
            2. List 3 customer pain points
            3. Return ONLY this JSON format (no other text):
            
            {
              "trends": [
                {"title": "Brief Title", "summary": "One sentence.", "url": "AI Knowledge Base"}
              ],
              "pain_points": [
                {"title": "Brief Title", "summary": "One sentence.", "url": "AI Knowledge Base"}
              ]
            }
            """

        try:
            result = agent_executor.invoke({"input": task})
            output = result.get("output", "")
            
            # Handle case where agent hit limits
            if "Agent stopped due to iteration limit" in output or "time limit" in output:
                print("⚠️ Agent hit execution limits, using minimal fallback data")
                logger.warning("Using fallback data - real analysis failed")
                data = {
                    "trends": [
                        {"title": "AI-First SaaS Design", "summary": "New SaaS products built with AI-native architecture from ground up.", "url": "Fallback Analysis"},
                        {"title": "Creator Economy Platforms", "summary": "Specialized platforms for creator monetization and audience building.", "url": "Fallback Analysis"},
                        {"title": "No-Code Enterprise Tools", "summary": "Enterprise adoption of no-code platforms for internal processes.", "url": "Fallback Analysis"}
                    ],
                    "pain_points": [
                        {"title": "SaaS Tool Sprawl", "summary": "Companies struggling to manage and integrate multiple SaaS subscriptions.", "url": "Fallback Analysis"},
                        {"title": "Creator Payment Delays", "summary": "Creators experiencing significant delays in platform revenue payments.", "url": "Fallback Analysis"},
                        {"title": "AI Implementation Costs", "summary": "Small businesses finding AI integration costs prohibitively expensive.", "url": "Fallback Analysis"}
                    ]
                }
            else:
                # More robust JSON extraction with fallback
                if not output or output.strip() == "":
                    print("⚠️ Agent returned empty output. Creating default structure.")
                    data = {"trends": [], "pain_points": []}
                else:
                    # Try to find JSON in the output
                    json_start = output.find('{')
                    json_end = output.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = output[json_start:json_end+1]
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as json_err:
                            print(f"⚠️ JSON parsing failed: {json_err}")
                            print(f"Raw output: {output[:200]}...")
                            
                            # Try to repair truncated JSON
                            try:
                                # Attempt to find the last complete object/array
                                repaired_json = json_str
                                
                                # If JSON ends abruptly, try to close it properly
                                if not repaired_json.rstrip().endswith('}'):
                                    # Count opening and closing braces to balance them
                                    open_braces = repaired_json.count('{')
                                    close_braces = repaired_json.count('}')
                                    
                                    # Remove any incomplete trailing content after last complete field
                                    lines = repaired_json.split('\n')
                                    complete_lines = []
                                    for line in lines:
                                        # Skip lines that look incomplete (missing quotes, colons, etc.)
                                        if ':' in line or line.strip() in ['{', '}', '[', ']'] or line.strip().endswith(','):
                                            complete_lines.append(line)
                                        elif line.strip() and not line.strip().startswith('"'):
                                            # Stop at incomplete line
                                            break
                                        else:
                                            complete_lines.append(line)
                                    
                                    repaired_json = '\n'.join(complete_lines)
                                    
                                    # Remove trailing comma if present
                                    repaired_json = repaired_json.rstrip().rstrip(',')
                                    
                                    # Add missing closing braces
                                    missing_braces = open_braces - close_braces
                                    for _ in range(missing_braces):
                                        if 'trends' in repaired_json and 'pain_points' not in repaired_json:
                                            repaired_json += '\n    }\n  ],\n  "pain_points": []\n}'
                                            break
                                        else:
                                            repaired_json += '\n}'
                                
                                print(f"🔧 Attempting to repair JSON...")
                                data = json.loads(repaired_json)
                                print(f"✅ Successfully repaired truncated JSON!")
                                
                            except json.JSONDecodeError:
                                print(f"🔧 JSON repair failed, using default structure")
                                # Create default structure if JSON repair fails
                                data = {"trends": [], "pain_points": []}
                    else:
                        print("⚠️ No JSON structure found in agent output.")
                        print(f"Raw output: {output[:200]}...")
                        data = {"trends": [], "pain_points": []}
        
        except Exception as e:
            error_type = type(e).__name__
            error_details = str(e)
            
            print(f"\n\n--- 🚨 A CRITICAL ERROR OCCURRED DURING AGENT EXECUTION 🚨 ---\n")
            print(f"Error Type: {error_type}")
            print(f"Error Details: {error_details}")
            
            # Specific handling for different error types
            if "RatelimitException" in error_type or "Ratelimit" in error_details:
                print(f"\n🔄 This is a rate limiting issue. Consider:")
                print("1. Waiting a few minutes before retrying")
                print("2. The system is using free models and search tools")
                print("3. Rate limits are normal for free services")
            elif "402" in error_details or "credits" in error_details.lower():
                print(f"\n💳 This is a credit/payment issue. Consider:")
                print("1. Check OpenRouter account balance")
                print("2. Verify API key has sufficient credits")
                print("3. Some models may require paid access")
            else:
                print(f"\nConsider checking:")
                print("1. OpenRouter API key validity")
                print("2. Model availability and quotas")
                print("3. Network connectivity")
                print("4. Search service availability")
            
            # Use fallback data even on error
            data = {
                "trends": [
                    {"title": "Analysis unavailable", "summary": "Fallback mode - real analysis required", "url": "System"}
                ],
                "pain_points": [
                    {"title": "Analysis unavailable", "summary": "Fallback mode - real analysis required", "url": "System"}
                ]
            }

    # Ensure data has required keys (for both test mode and regular execution)
    data.setdefault("trends", [])
    data.setdefault("pain_points", [])

    try:
        print("\n--- Processing and Storing Agent Results ---")
        for trend in data.get("trends", []):
            if isinstance(trend, dict) and trend.get('title'):
                db_payload = {
                    'type': 'trend',
                    'source_url': trend.get('url', ''),
                    'processed_insights_path': f"{trend.get('title')} - {trend.get('summary', '')}"
                }
                supabase.table('data_sources').insert(db_payload).execute()
                print(f"✅ STORED TREND: {trend.get('title')}")

        for pain in data.get("pain_points", []):
            if isinstance(pain, dict) and pain.get('title'):
                db_payload = {
                    'type': 'pain_point',
                    'source_url': pain.get('url', ''),
                    'processed_insights_path': f"{pain.get('title')} - {pain.get('summary', '')}"
                }
                supabase.table('data_sources').insert(db_payload).execute()
                print(f"✅ STORED PAIN POINT: {pain.get('title')}")
    except Exception as db_error:
        print(f"⚠️ Database storage failed: {db_error}")
        print("Data collection completed but storage failed")

    print("\n--- Data Ingestion Run Complete ---")

if __name__ == "__main__":
    run_market_analysis()
