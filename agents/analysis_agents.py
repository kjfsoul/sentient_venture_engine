# sentient_venture_engine/agents/analysis_agents.py
# Code Analysis Agents for GitHub Repository Intelligence

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from security.api_key_manager import get_secret
from supabase import create_client
from langchain_openai import ChatOpenAI

class GitHubCodeAnalysisAgent:
    """Agent for analyzing GitHub repositories to identify technology trends and market insights."""
    
    def __init__(self):
        self.github_token = get_secret('GITHUB_TOKEN')
        self.supabase = create_client(get_secret('SUPABASE_URL'), get_secret('SUPABASE_KEY'))
        self.llm = self._initialize_llm()
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def _initialize_llm(self):
        """Initialize LLM for code analysis."""
        api_key = get_secret("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model="microsoft/phi-3-mini-128k-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
            default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE-CodeAnalysis"}
        )
    
    def search_trending_repositories(self, query: str = "created:>2024-01-01", limit: int = 20) -> List[Dict]:
        """Search for trending repositories based on criteria."""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': f'{query} stars:>100',
            'sort': 'stars',
            'order': 'desc',
            'per_page': limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            print(f"Error searching repositories: {e}")
            return []
    
    def analyze_repository_technologies(self, repo_data: Dict) -> Dict:
        """Analyze a repository to extract technology insights."""
        try:
            # Get language statistics
            languages_url = repo_data.get('languages_url')
            languages_response = requests.get(languages_url, headers=self.headers)
            languages = languages_response.json() if languages_response.status_code == 200 else {}
            
            # Get repository topics/tags
            topics = repo_data.get('topics', [])
            
            # Analyze with LLM
            analysis_prompt = f"""
            Analyze this GitHub repository for market intelligence:
            
            Repository: {repo_data.get('name')}
            Description: {repo_data.get('description', 'No description')}
            Stars: {repo_data.get('stargazers_count', 0)}
            Languages: {list(languages.keys())}
            Topics: {topics}
            
            Extract market insights in JSON format:
            {{
                "technology_trend": "Brief trend name",
                "market_signal": "What this indicates about market direction",
                "adoption_level": "emerging|growing|mature",
                "business_opportunity": "Potential business opportunity"
            }}
            
            Return only valid JSON.
            """
            
            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            try:
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis = json.loads(analysis_text[json_start:json_end])
                else:
                    analysis = self._create_fallback_analysis(repo_data, languages, topics)
            except json.JSONDecodeError:
                analysis = self._create_fallback_analysis(repo_data, languages, topics)
            
            # Add metadata
            analysis.update({
                'repository_name': repo_data.get('name'),
                'repository_url': repo_data.get('html_url'),
                'stars': repo_data.get('stargazers_count', 0),
                'primary_language': max(languages.keys(), key=languages.get) if languages else 'Unknown',
                'languages': list(languages.keys()),
                'topics': topics,
                'created_at': repo_data.get('created_at'),
                'updated_at': repo_data.get('updated_at')
            })
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing repository {repo_data.get('name')}: {e}")
            return self._create_fallback_analysis(repo_data, {}, [])
    
    def _create_fallback_analysis(self, repo_data: Dict, languages: Dict, topics: List) -> Dict:
        """Create fallback analysis when LLM fails."""
        primary_lang = max(languages.keys(), key=languages.get) if languages else 'Unknown'
        
        # Simple heuristics for technology trends
        if 'ai' in str(topics).lower() or 'ml' in str(topics).lower():
            trend = "AI/ML Integration"
            signal = "Growing AI adoption in software development"
        elif 'web3' in str(topics).lower() or 'blockchain' in str(topics).lower():
            trend = "Web3 Development"
            signal = "Blockchain technology adoption"
        elif primary_lang in ['TypeScript', 'JavaScript']:
            trend = "Modern Web Development"
            signal = "JavaScript ecosystem evolution"
        elif primary_lang == 'Python':
            trend = "Python Ecosystem Growth"
            signal = "Python's expanding use cases"
        else:
            trend = f"{primary_lang} Development"
            signal = f"Activity in {primary_lang} ecosystem"
        
        return {
            'technology_trend': trend,
            'market_signal': signal,
            'adoption_level': 'growing',
            'business_opportunity': f"Tools and services for {trend.lower()}"
        }
    
    def store_analysis_results(self, analyses: List[Dict]):
        """Store analysis results in Supabase."""
        for analysis in analyses:
            try:
                # Store in code_analysis table
                db_payload = {
                    'type': 'code_trend',
                    'source_url': analysis.get('repository_url', ''),
                    'processed_insights_path': f"{analysis.get('technology_trend')} - {analysis.get('market_signal')}",
                    'metadata': json.dumps({
                        'repository': analysis.get('repository_name'),
                        'stars': analysis.get('stars'),
                        'languages': analysis.get('languages'),
                        'topics': analysis.get('topics'),
                        'adoption_level': analysis.get('adoption_level'),
                        'business_opportunity': analysis.get('business_opportunity')
                    })
                }
                
                result = self.supabase.table('data_sources').insert(db_payload).execute()
                print(f"✅ STORED CODE TREND: {analysis.get('technology_trend')}")
                
            except Exception as e:
                print(f"⚠️ Failed to store analysis for {analysis.get('repository_name')}: {e}")
    
    def run_code_analysis(self, search_queries: List[str] = None) -> Dict:
        """Run comprehensive code analysis for market intelligence."""
        if search_queries is None:
            search_queries = [
                "created:>2024-01-01 topic:ai",
                "created:>2024-01-01 topic:saas", 
                "created:>2024-01-01 language:python",
                "created:>2024-01-01 topic:startup"
            ]
        
        all_analyses = []
        
        for query in search_queries:
            print(f"🔍 Searching repositories: {query}")
            repos = self.search_trending_repositories(query, limit=5)
            
            for repo in repos:
                analysis = self.analyze_repository_technologies(repo)
                if analysis:
                    all_analyses.append(analysis)
        
        # Store results
        if all_analyses:
            self.store_analysis_results(all_analyses)
        
        print(f"📊 Code Analysis Complete: Processed {len(all_analyses)} repositories")
        return {
            'total_analyzed': len(all_analyses),
            'trends_identified': len(set(a.get('technology_trend') for a in all_analyses)),
            'analyses': all_analyses[:5]  # Return sample for verification
        }

def run_code_intelligence():
    """Main function to run code intelligence analysis."""
    print("🚀 Starting GitHub Code Analysis Agent...")
    
    try:
        agent = GitHubCodeAnalysisAgent()
        results = agent.run_code_analysis()
        
        print("\n--- Code Intelligence Results ---")
        print(f"Repositories analyzed: {results['total_analyzed']}")
        print(f"Unique trends identified: {results['trends_identified']}")
        print("\nSample trends:")
        for analysis in results['analyses']:
            print(f"  • {analysis.get('technology_trend')}: {analysis.get('market_signal')}")
        
        print("\n✅ Code Analysis Complete")
        return results
        
    except Exception as e:
        print(f"❌ Code analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    run_code_intelligence()
