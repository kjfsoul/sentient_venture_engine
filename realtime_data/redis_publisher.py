# sentient_venture_engine/realtime_data/redis_publisher.py
# Real-time Data Publisher for Market Intelligence Events

import redis
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from security.api_key_manager import get_secret

class MarketIntelPublisher:
    """Publisher for real-time market intelligence events."""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize Redis publisher."""
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def publish_market_event(self, channel: str, event_data: Dict):
        """Publish a market intelligence event to Redis channel."""
        try:
            # Add timestamp and metadata
            enriched_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'channel': channel,
                'event_id': f"{channel}_{int(time.time())}",
                **event_data
            }
            
            # Publish to Redis
            message = json.dumps(enriched_event)
            subscribers = self.redis_client.publish(channel, message)
            
            print(f"📡 Published to {channel}: {event_data.get('title', 'Event')} ({subscribers} subscribers)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish event: {e}")
            return False
    
    def monitor_news_feeds(self, keywords: List[str] = None):
        """Monitor news feeds for market intelligence (simulation)."""
        if keywords is None:
            keywords = ['AI', 'SaaS', 'startup', 'venture capital', 'technology']
        
        # Simulate real-time news events
        sample_events = [
            {
                'title': 'Major SaaS Company Announces AI Integration',
                'category': 'saas_trends',
                'sentiment': 'positive',
                'impact_score': 8.5,
                'source': 'TechCrunch',
                'keywords': ['SaaS', 'AI', 'integration']
            },
            {
                'title': 'Venture Capital Funding Reaches New High',
                'category': 'funding_trends', 
                'sentiment': 'positive',
                'impact_score': 9.2,
                'source': 'VentureBeat',
                'keywords': ['venture capital', 'funding', 'startup']
            },
            {
                'title': 'New No-Code Platform Gains Million Users',
                'category': 'product_trends',
                'sentiment': 'positive', 
                'impact_score': 7.8,
                'source': 'ProductHunt',
                'keywords': ['no-code', 'platform', 'users']
            }
        ]
        
        for event in sample_events:
            # Determine appropriate channel based on category
            if event['category'] == 'saas_trends':
                channel = 'market:saas'
            elif event['category'] == 'funding_trends':
                channel = 'market:funding'
            else:
                channel = 'market:general'
            
            self.publish_market_event(channel, event)
            time.sleep(2)  # Simulate real-time spacing
    
    def monitor_github_activity(self):
        """Monitor GitHub for trending repositories (real API integration)."""
        try:
            github_token = get_secret('GITHUB_TOKEN')
            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Get trending repositories from today
            url = "https://api.github.com/search/repositories"
            params = {
                'q': f'created:>{datetime.now().strftime("%Y-%m-%d")} stars:>10',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 5
            }
            
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                repos = response.json().get('items', [])
                
                for repo in repos:
                    event = {
                        'title': f"Trending Repository: {repo['name']}",
                        'repository': repo['name'],
                        'stars': repo['stargazers_count'],
                        'language': repo.get('language', 'Unknown'),
                        'description': repo.get('description', ''),
                        'url': repo['html_url'],
                        'category': 'github_trending',
                        'impact_score': min(repo['stargazers_count'] / 10, 10)
                    }
                    
                    self.publish_market_event('market:github', event)
            
        except Exception as e:
            print(f"❌ GitHub monitoring error: {e}")
    
    def simulate_real_time_monitoring(self, duration_minutes: int = 5):
        """Simulate real-time market monitoring for testing."""
        print(f"🔄 Starting {duration_minutes}-minute real-time monitoring simulation...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Alternate between different monitoring types
            cycle_time = int(time.time()) % 60
            
            if cycle_time % 20 == 0:
                self.monitor_news_feeds()
            elif cycle_time % 30 == 0:
                self.monitor_github_activity()
            
            time.sleep(10)  # Check every 10 seconds
        
        print("✅ Real-time monitoring simulation complete")

def main():
    """Main function to run the publisher."""
    print("🚀 Starting Market Intelligence Publisher...")
    
    try:
        publisher = MarketIntelPublisher()
        
        # Check if we're in simulation mode
        simulation_mode = os.getenv('REDIS_SIMULATION', 'true').lower() == 'true'
        
        if simulation_mode:
            print("📊 Running in simulation mode...")
            publisher.simulate_real_time_monitoring(duration_minutes=2)
        else:
            print("📡 Running real-time monitoring...")
            publisher.monitor_news_feeds()
            publisher.monitor_github_activity()
    
    except Exception as e:
        print(f"❌ Publisher failed: {e}")

if __name__ == '__main__':
    main()
