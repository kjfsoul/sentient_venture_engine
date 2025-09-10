#!/usr/bin/env python3
"""
Code Analysis Agent for GitHub Repository Intelligence
Task 1.1.2: MarketIntelAgents for Code Analysis

Integrates with:
- GitHub API for repository analysis
- Qwen 3 Coder, Deepseek, Claude Code Max for AI code analysis
- Trend identification in open-source projects
- Rate limiting and failure handling
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from collections import Counter
from functools import wraps

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

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
    'openrouter': {'requests_per_minute': 15, 'burst_limit': 8},
    'together': {'requests_per_minute': 10, 'burst_limit': 5}
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

# Rate limiting configuration for GitHub API
GITHUB_RATE_LIMIT = {'requests_per_hour': 5000, 'burst_limit': 100}
github_request_times = []

def github_rate_limit():
    """Rate limiting decorator for GitHub API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove requests older than 1 hour
            global github_request_times
            github_request_times = [
                req_time for req_time in github_request_times 
                if now - req_time < 3600
            ]
            
            # Check if we're at the rate limit
            if len(github_request_times) >= GITHUB_RATE_LIMIT['requests_per_hour']:
                # Calculate wait time
                oldest_request = min(github_request_times)
                wait_time = 3600 - (now - oldest_request)
                if wait_time > 0:
                    logger.warning(f"⏳ GitHub rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
            
            # Record this request
            github_request_times.append(now)
            
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
                    if 'rate' in error_str or '429' in error_str or 'secondary' in error_str:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + (0.1 * (attempt + 1))
                        jitter = 0.1 * delay * (2 * (hash(str(attempt)) % 1000) / 1000 - 1)
                        total_delay = max(0, delay + jitter)
                        
                        logger.warning(f"⚠️ GitHub rate limit hit on attempt {attempt + 1}, backing off for {total_delay:.2f} seconds")
                        time.sleep(total_delay)
                    else:
                        # Non-rate limit error, re-raise immediately
                        raise e
            return None
        return wrapper
    return decorator

@dataclass
class CodeRepository:
    """Represents a code repository for analysis"""
    name: str
    owner: str
    url: str
    description: str
    language: str
    stars: int
    forks: int
    created_at: datetime
    updated_at: datetime
    topics: List[str]
    readme_content: str

@dataclass
class CodeAnalysis:
    """Results from code repository analysis"""
    repository_name: str
    owner: str
    technological_trends: List[str]
    emerging_patterns: List[str]
    language_adoption: Dict[str, int]
    framework_usage: List[str]
    ai_insights: Dict[str, Any]
    confidence_scores: Dict[str, float]
    analysis_timestamp: datetime

class CodeAnalysisAgent:
    """Agent for analyzing GitHub repositories for technological trends"""
    
    def __init__(self):
        self.github_token = get_secret('GITHUB_TOKEN')
        self.openrouter_key = get_secret('OPENROUTER_API_KEY')
        self.together_key = get_secret('TOGETHER_API_KEY')  # Additional provider
        
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
        
        # AI code analysis models (using available providers)
        self.code_analysis_models = [
            "openrouter/qwen/qwen-coder-2.5-7b",  # Qwen 3 Coder via OpenRouter
            "openrouter/deepseek/deepseek-coder-33b",  # Deepseek via OpenRouter
            "openrouter/anthropic/claude-3-5-sonnet",  # Claude Code Max via OpenRouter
            "together/Qwen/Qwen2-72B-Instruct",  # Qwen via Together.ai
            "together/deepseek-ai/DeepSeek-Coder-V2-Instruct",  # Deepseek via Together.ai
        ]
        
        # Headers for GitHub API
        self.github_headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    @github_rate_limit()
    @exponential_backoff(max_retries=3)
    def search_trending_repositories(self, days_back: int = 7, limit: int = 20) -> List[CodeRepository]:
        """Search for trending repositories on GitHub"""
        try:
            # Calculate date for search
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Search for recently updated repositories with high star count
            search_url = "https://api.github.com/search/repositories"
            params = {
                'q': f'created:>{since_date} stars:>10',
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            response = requests.get(search_url, headers=self.github_headers, params=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                raise Exception("GitHub rate limit exceeded")
            
            response.raise_for_status()
            
            repositories = []
            for repo_data in response.json().get('items', []):
                # Get README content
                readme_content = self._get_repository_readme(repo_data['owner']['login'], repo_data['name'])
                
                repo = CodeRepository(
                    name=repo_data['name'],
                    owner=repo_data['owner']['login'],
                    url=repo_data['html_url'],
                    description=repo_data.get('description', ''),
                    language=repo_data.get('language', 'Unknown'),
                    stars=repo_data['stargazers_count'],
                    forks=repo_data['forks_count'],
                    created_at=datetime.strptime(repo_data['created_at'], '%Y-%m-%dT%H:%M:%SZ'),
                    updated_at=datetime.strptime(repo_data['updated_at'], '%Y-%m-%dT%H:%M:%SZ'),
                    topics=repo_data.get('topics', []),
                    readme_content=readme_content
                )
                repositories.append(repo)
            
            logger.info(f"✅ Found {len(repositories)} trending repositories")
            return repositories
            
        except Exception as e:
            logger.error(f"❌ Failed to search trending repositories: {e}")
            # Re-raise rate limit errors for backoff
            if "rate" in str(e).lower() or "429" in str(e):
                raise e
            return []

    @github_rate_limit()
    @exponential_backoff(max_retries=3)
    def _get_repository_readme(self, owner: str, repo_name: str) -> str:
        """Get README content from a repository"""
        try:
            readme_url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
            response = requests.get(readme_url, headers=self.github_headers, timeout=15)
            
            # Handle rate limiting
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                raise Exception("GitHub rate limit exceeded")
            
            response.raise_for_status()
            
            readme_data = response.json()
            # Get the actual content (base64 encoded)
            import base64
            content = base64.b64decode(readme_data['content']).decode('utf-8')
            return content[:2000]  # Limit content size
        except Exception as e:
            logger.warning(f"⚠️ Failed to get README for {owner}/{repo_name}: {e}")
            # Re-raise rate limit errors for backoff
            if "rate" in str(e).lower() or "429" in str(e):
                raise e
            return ""

    @rate_limit('openrouter')
    @exponential_backoff(max_retries=3)
    def analyze_code_with_ai_model(self, repository: CodeRepository) -> Dict[str, Any]:
        """Analyze repository code using AI code analysis models"""
        analysis_prompt = f"""
        Analyze this GitHub repository for technological trends and patterns:
        
        Repository: {repository.name}
        Owner: {repository.owner}
        Description: {repository.description}
        Primary Language: {repository.language}
        Stars: {repository.stars}
        Topics: {', '.join(repository.topics)}
        
        README Content:
        {repository.readme_content[:1000]}...
        
        Provide analysis in JSON format:
        {{
            "technological_trends": ["list of identified tech trends"],
            "emerging_patterns": ["list of emerging coding patterns"],
            "language_adoption": {{"language": count}},
            "framework_usage": ["list of frameworks/libraries used"],
            "confidence_scores": {{"trends": 0.9, "patterns": 0.8}}
        }}
        """
        
        # Try different providers
        providers = [
            {
                'name': 'openrouter',
                'api_key': self.openrouter_key,
                'base_url': 'https://openrouter.ai/api/v1/chat/completions'
            },
            {
                'name': 'together',
                'api_key': self.together_key,
                'base_url': 'https://api.together.xyz/v1/chat/completions'
            }
        ]
        
        for provider in providers:
            if not provider['api_key']:
                continue
                
            for model in self.code_analysis_models:
                # Filter models by provider
                if provider['name'] == 'openrouter' and 'openrouter/' not in model:
                    continue
                if provider['name'] == 'together' and 'together/' not in model:
                    continue
                    
                try:
                    logger.info(f"🔍 Analyzing {repository.name} with {provider['name']}/{model}")
                    
                    headers = {
                        "Authorization": f"Bearer {provider['api_key']}",
                        "Content-Type": "application/json"
                    }
                    
                    # Adjust model name for the API call
                    api_model = model.replace(f"{provider['name']}/", "")
                    
                    payload = {
                        "model": api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": analysis_prompt
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                    
                    response = requests.post(
                        provider['base_url'],
                        headers=headers,
                        json=payload,
                        timeout=45
                    )
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        raise Exception("Rate limit exceeded")
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Try to parse as JSON
                        try:
                            analysis = json.loads(content)
                            analysis['model_used'] = f"{provider['name']}/{model}"
                            return analysis
                        except json.JSONDecodeError:
                            # If not JSON, try to extract JSON from text
                            json_start = content.find('{')
                            json_end = content.rfind('}')
                            if json_start != -1 and json_end != -1 and json_end > json_start:
                                json_str = content[json_start:json_end+1]
                                try:
                                    analysis = json.loads(json_str)
                                    analysis['model_used'] = f"{provider['name']}/{model}"
                                    return analysis
                                except json.JSONDecodeError:
                                    continue
                            continue
                    
                except Exception as e:
                    logger.warning(f"Model {provider['name']}/{model} failed for {repository.name}: {e}")
                    # Re-raise rate limit errors for backoff
                    if "rate" in str(e).lower() or "429" in str(e):
                        raise e
                    continue
        
        return {"error": "All code analysis models failed"}

    def analyze_repositories(self, repositories: List[CodeRepository]) -> List[CodeAnalysis]:
        """Analyze multiple repositories for technological trends"""
        analyses = []
        
        for repo in repositories:
            try:
                # Analyze with AI model
                ai_result = self.analyze_code_with_ai_model(repo)
                
                if "error" not in ai_result:
                    analysis = CodeAnalysis(
                        repository_name=repo.name,
                        owner=repo.owner,
                        technological_trends=ai_result.get('technological_trends', []),
                        emerging_patterns=ai_result.get('emerging_patterns', []),
                        language_adoption=ai_result.get('language_adoption', {}),
                        framework_usage=ai_result.get('framework_usage', []),
                        ai_insights=ai_result,
                        confidence_scores=ai_result.get('confidence_scores', {}),
                        analysis_timestamp=datetime.now()
                    )
                    analyses.append(analysis)
                    logger.info(f"✅ Analyzed: {repo.name}")
                else:
                    logger.error(f"❌ Failed to analyze: {repo.name}")
                
                # Rate limiting
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error analyzing {repo.name}: {e}")
                continue
        
        return analyses

    def generate_code_insights_report(self, analyses: List[CodeAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive insights from code analyses"""
        if not analyses:
            return {"error": "No analyses provided"}
        
        # Aggregate trends
        all_trends = []
        all_patterns = []
        all_frameworks = []
        language_counts = Counter()
        confidence_scores = []
        
        for analysis in analyses:
            all_trends.extend(analysis.technological_trends)
            all_patterns.extend(analysis.emerging_patterns)
            all_frameworks.extend(analysis.framework_usage)
            language_counts.update(analysis.language_adoption)
            confidence_scores.extend(analysis.confidence_scores.values())
        
        insights = {
            'analysis_summary': {
                'total_repositories_analyzed': len(analyses),
                'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'technological_trends': dict(Counter(all_trends).most_common(10)),
            'emerging_patterns': dict(Counter(all_patterns).most_common(10)),
            'framework_adoption': dict(Counter(all_frameworks).most_common(10)),
            'language_popularity': dict(language_counts.most_common(10)),
            'market_opportunities': self._identify_code_market_opportunities(analyses)
        }
        
        return insights

    def _identify_code_market_opportunities(self, analyses: List[CodeAnalysis]) -> List[str]:
        """Identify potential market opportunities from code trends"""
        opportunities = []
        
        # Analyze patterns to suggest opportunities
        all_elements = []
        for analysis in analyses:
            all_elements.extend(analysis.technological_trends)
            all_elements.extend(analysis.framework_usage)
        
        trend_counts = Counter(all_elements)
        
        # Generate opportunity insights
        for trend, count in trend_counts.most_common(5):
            if count >= 2:  # Threshold for significance
                opportunities.append(f"Growing adoption of {trend} - {count} repositories detected")
        
        return opportunities

    def store_code_intelligence(self, insights: Dict[str, Any]) -> bool:
        """Store code intelligence insights in Supabase"""
        if not self.supabase:
            logger.warning("Supabase not available - insights not stored")
            return False
        
        try:
            # Prepare data for storage
            storage_data = {
                'analysis_type': 'code_intelligence',
                'insights': insights,
                'timestamp': datetime.now().isoformat(),
                'source': 'code_analysis_agent'
            }
            
            # Store in market_intelligence table
            result = self.supabase.table('market_intelligence').insert(storage_data).execute()
            
            if result.data:
                logger.info("✅ Code intelligence stored successfully")
                return True
            else:
                logger.error("❌ Failed to store code intelligence")
                return False
                
        except Exception as e:
            logger.error(f"Error storing code intelligence: {e}")
            return False

    def run_code_intelligence_analysis(self) -> Dict[str, Any]:
        """Main execution method for code intelligence gathering"""
        logger.info("🚀 Starting Code Intelligence Analysis")
        
        try:
            # Step 1: Search for trending repositories
            logger.info("🔍 Searching for trending repositories...")
            repositories = self.search_trending_repositories(days_back=7, limit=15)
            
            if not repositories:
                return {"error": "No repositories found"}
            
            # Step 2: Analyze repositories
            logger.info(f"🧠 Analyzing {len(repositories)} repositories...")
            analyses = self.analyze_repositories(repositories)
            
            if not analyses:
                return {"error": "No successful analyses"}
            
            # Step 3: Generate insights report
            logger.info("📊 Generating code insights report...")
            insights = self.generate_code_insights_report(analyses)
            
            # Step 4: Store results
            logger.info("💾 Storing code intelligence...")
            stored = self.store_code_intelligence(insights)
            
            # Return results
            final_results = {
                'success': True,
                'insights': insights,
                'repositories_analyzed': len(repositories),
                'successful_analyses': len(analyses),
                'stored_successfully': stored,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Code Intelligence Analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Code Intelligence Analysis failed: {e}")
            return {"error": str(e), "success": False}

def main():
    """Main execution function"""
    print("💻 Starting Code Analysis Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = CodeAnalysisAgent()
    
    # Run analysis
    results = agent.run_code_intelligence_analysis()
    
    # Display results
    if results.get('success'):
        print("\n✅ CODE INTELLIGENCE ANALYSIS COMPLETE")
        print(f"📊 Repositories analyzed: {results['repositories_analyzed']}")
        print(f"🔍 Successful analyses: {results['successful_analyses']}")
        print(f"💾 Data stored: {results['stored_successfully']}")
        
        insights = results['insights']
        print(f"\n📈 CODE TRENDS DISCOVERED:")
        print(f"🔥 Top tech trends: {list(insights.get('technological_trends', {}).keys())[:3]}")
        print(f"🚀 Framework adoption: {list(insights.get('framework_adoption', {}).keys())[:3]}")
        print(f"🌐 Language popularity: {list(insights.get('language_popularity', {}).keys())[:3]}")
        
        if insights.get('market_opportunities'):
            print(f"\n💡 MARKET OPPORTUNITIES:")
            for opportunity in insights['market_opportunities'][:3]:
                print(f"   • {opportunity}")
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
