# sentient_venture_engine/realtime_data/redis_consumer.py
# Real-time Data Consumer for Market Intelligence Events

import redis
import json
import time
from datetime import datetime
from typing import Dict, List, Callable
import os
from pathlib import Path
import threading

# Add project root to Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from security.api_key_manager import get_secret
from supabase import create_client
from langchain_openai import ChatOpenAI

class MarketIntelConsumer:
    """Consumer for processing real-time market intelligence events."""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize Redis consumer and dependencies."""
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
            
            # Initialize Supabase for data storage
            self.supabase = create_client(get_secret('SUPABASE_URL'), get_secret('SUPABASE_KEY'))
            
            # Initialize LLM for event analysis
            self.llm = self._initialize_llm()
            
            # Event counters
            self.events_processed = 0
            self.alerts_triggered = 0
            
        except Exception as e:
            print(f"❌ Failed to initialize consumer: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM for event analysis."""
        try:
            api_key = get_secret("OPENROUTER_API_KEY")
            return ChatOpenAI(
                model="microsoft/phi-3-mini-128k-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=0.2,
                max_tokens=512,
                default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE-RealTime"}
            )
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            return None
    
    def analyze_event_significance(self, event_data: Dict) -> Dict:
        """Analyze event significance using LLM."""
        if not self.llm:
            return self._create_fallback_analysis(event_data)
        
        try:
            analysis_prompt = f"""
            Analyze this real-time market event for significance:
            
            Title: {event_data.get('title', 'Unknown')}
            Category: {event_data.get('category', 'Unknown')}
            Impact Score: {event_data.get('impact_score', 0)}
            Keywords: {event_data.get('keywords', [])}
            
            Determine:
            1. Market significance (low/medium/high)
            2. Alert priority (none/low/medium/high)
            3. Business implications
            
            Return JSON:
            {{
                "significance": "low|medium|high",
                "alert_priority": "none|low|medium|high",
                "business_implications": "Brief description",
                "recommended_action": "What should be done"
            }}
            """
            
            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(analysis_text[json_start:json_end])
            else:
                return self._create_fallback_analysis(event_data)
                
        except Exception as e:
            print(f"⚠️ Event analysis failed: {e}")
            return self._create_fallback_analysis(event_data)
    
    def _create_fallback_analysis(self, event_data: Dict) -> Dict:
        """Create fallback analysis when LLM fails."""
        impact_score = event_data.get('impact_score', 0)
        
        if impact_score >= 8:
            significance = "high"
            alert_priority = "high"
        elif impact_score >= 6:
            significance = "medium"
            alert_priority = "medium"
        else:
            significance = "low"
            alert_priority = "low"
        
        return {
            'significance': significance,
            'alert_priority': alert_priority,
            'business_implications': f"Event with {impact_score} impact score in {event_data.get('category', 'unknown')} category",
            'recommended_action': 'Monitor for related developments'
        }
    
    def store_processed_event(self, event_data: Dict, analysis: Dict):
        """Store processed event in Supabase."""
        try:
            db_payload = {
                'type': 'realtime_event',
                'source_url': event_data.get('url', ''),
                'processed_insights_path': f"{event_data.get('title')} - {analysis.get('business_implications')}",
                'metadata': json.dumps({
                    'original_event': event_data,
                    'analysis': analysis,
                    'processed_at': datetime.utcnow().isoformat(),
                    'alert_priority': analysis.get('alert_priority'),
                    'significance': analysis.get('significance')
                })
            }
            
            result = self.supabase.table('data_sources').insert(db_payload).execute()
            print(f"✅ STORED REALTIME EVENT: {event_data.get('title')}")
            
        except Exception as e:
            print(f"❌ Failed to store event: {e}")
    
    def trigger_alert(self, event_data: Dict, analysis: Dict):
        """Trigger alert for high-priority events."""
        alert_priority = analysis.get('alert_priority', 'low')
        
        if alert_priority in ['high', 'medium']:
            alert_message = {
                'alert_type': 'market_intelligence',
                'priority': alert_priority,
                'title': event_data.get('title'),
                'summary': analysis.get('business_implications'),
                'recommended_action': analysis.get('recommended_action'),
                'timestamp': datetime.utcnow().isoformat(),
                'source_channel': event_data.get('channel')
            }
            
            # Publish alert to special alert channel
            alert_json = json.dumps(alert_message)
            self.redis_client.publish('alerts:market', alert_json)
            
            print(f"🚨 ALERT TRIGGERED ({alert_priority}): {event_data.get('title')}")
            self.alerts_triggered += 1
    
    def process_event(self, channel: str, event_data: Dict):
        """Process a single market intelligence event."""
        try:
            print(f"⚙️ Processing event from {channel}: {event_data.get('title', 'Unknown')}")
            
            # Analyze event significance
            analysis = self.analyze_event_significance(event_data)
            
            # Store processed event
            self.store_processed_event(event_data, analysis)
            
            # Check if alert should be triggered
            self.trigger_alert(event_data, analysis)
            
            self.events_processed += 1
            
        except Exception as e:
            print(f"❌ Error processing event: {e}")
    
    def message_handler(self, message):
        """Handle incoming Redis messages."""
        try:
            if message['type'] == 'message':
                channel = message['channel']
                data = json.loads(message['data'])
                self.process_event(channel, data)
        except Exception as e:
            print(f"❌ Message handling error: {e}")
    
    def subscribe_to_channels(self, channels: List[str]):
        """Subscribe to Redis channels and process messages."""
        print(f"📡 Subscribing to channels: {', '.join(channels)}")
        
        pubsub = self.redis_client.pubsub()
        
        # Subscribe to channels
        for channel in channels:
            pubsub.subscribe(channel)
        
        print("✅ Subscription active, waiting for messages...")
        
        try:
            for message in pubsub.listen():
                self.message_handler(message)
        except KeyboardInterrupt:
            print("\n🚪 Shutting down consumer...")
        finally:
            pubsub.close()
            print(f"📊 Final stats - Events processed: {self.events_processed}, Alerts triggered: {self.alerts_triggered}")
    
    def run_consumer(self, duration_minutes: int = None):
        """Run the consumer for a specified duration or indefinitely."""
        # Default channels to monitor
        channels = [
            'market:saas',
            'market:funding', 
            'market:github',
            'market:general'
        ]
        
        if duration_minutes:
            print(f"🔄 Running consumer for {duration_minutes} minutes...")
            
            # Run in thread with timeout
            def timed_consumer():
                self.subscribe_to_channels(channels)
            
            consumer_thread = threading.Thread(target=timed_consumer)
            consumer_thread.daemon = True
            consumer_thread.start()
            
            # Wait for specified duration
            time.sleep(duration_minutes * 60)
            print(f"⏰ {duration_minutes} minutes elapsed, stopping consumer...")
            
        else:
            print("🔄 Running consumer indefinitely...")
            self.subscribe_to_channels(channels)

def main():
    """Main function to run the consumer."""
    print("🚀 Starting Market Intelligence Consumer...")
    
    try:
        consumer = MarketIntelConsumer()
        
        # Check if we're in test mode
        test_duration = os.getenv('CONSUMER_TEST_MINUTES')
        if test_duration:
            consumer.run_consumer(duration_minutes=int(test_duration))
        else:
            consumer.run_consumer()  # Run indefinitely
    
    except Exception as e:
        print(f"❌ Consumer failed: {e}")

if __name__ == '__main__':
    main()
