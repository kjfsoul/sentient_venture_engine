#!/usr/bin/env python3
"""
Groq Cloud LLM Integration for sentient_venture_engine
High-speed inference with free tier models including Llama-3 8B/70B

Features:
- Ultra-fast inference (fastest LLM provider)
- Free tier access to Llama-3 models
- Built-in rate limiting for free tier
- Cost-free operation
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GroqResponse:
    """Structured response from Groq API"""
    content: str
    model_used: str
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None

class GroqProvider:
    """
    Groq Cloud integration for sentient_venture_engine
    
    Features:
    - Ultra-fast inference (10x faster than traditional providers)
    - Free tier Llama-3 models
    - Built-in rate limiting
    - Cost-free operation
    """
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in environment variables")
            
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Free tier models (high-speed inference)
        self.free_models = [
            "llama3-8b-8192",          # Llama-3 8B (fast, efficient)
            "llama3-70b-8192",         # Llama-3 70B (high quality)
            "mixtral-8x7b-32768",      # Mixtral 8x7B (good balance)
            "gemma-7b-it",             # Gemma 7B (Google model)
        ]
        
        # Rate limiting for free tier (conservative limits)
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
        logger.info("⚡ Groq LLM Provider initialized (ultra-fast inference)")

    def _apply_rate_limit(self):
        """Apply rate limiting for free tier"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def test_connection(self) -> bool:
        """Test connection to Groq API"""
        if not self.api_key:
            logger.error("❌ Missing Groq API key")
            return False
            
        try:
            # Test with fastest model
            response = self.get_completion(
                messages=[{"role": "user", "content": "Say 'OK' if you're working."}],
                model="llama3-8b-8192",
                max_tokens=10
            )
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"❌ Groq connection test failed: {e}")
            return False

    def get_completion(self, 
                      messages: List[Dict[str, str]], 
                      model: str = "llama3-8b-8192",
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      timeout: int = 30) -> GroqResponse:
        """
        Get completion from Groq API with ultra-fast inference
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to fastest free model)
            max_tokens: Maximum tokens to generate (free tier limit: 8192)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Ensure we're using a free model
            if model not in self.free_models:
                logger.warning(f"Model {model} not in free tier, using default")
                model = self.free_models[0]
            
            # Limit tokens for free tier
            max_tokens = min(max_tokens, 1024)  # Conservative limit for free tier
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            start_time = time.time()
            logger.info(f"🔄 Requesting Groq completion with model: {model}")
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response content
                content = ""
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0].get('message', {}).get('content', '')
                
                # Extract usage info
                usage = data.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
                
                logger.info(f"✅ Groq response: {len(content)} chars, {tokens_used} tokens in {response_time:.2f}s")
                
                return GroqResponse(
                    content=content,
                    model_used=model,
                    success=True,
                    tokens_used=tokens_used,
                    response_time=response_time
                )
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Groq request failed: {error_msg}")
                
                return GroqResponse(
                    content="",
                    model_used=model,
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Request exception: {str(e)}"
            logger.error(f"❌ Groq request exception: {error_msg}")
            
            return GroqResponse(
                content="",
                model_used=model,
                success=False,
                error=error_msg
            )

    def market_analysis_request(self, 
                              prompt: str, 
                              model: str = "llama3-70b-8192",  # Use larger model for analysis
                              analysis_type: str = "market_intelligence") -> Dict[str, Any]:
        """
        Specialized market analysis request using Groq's high-speed inference
        """
        system_message = """You are a senior market intelligence analyst specializing in venture strategy and business opportunity identification. 
        Provide structured, actionable insights based on current market trends, competitive landscapes, and emerging opportunities.
        Format your response as valid JSON with clear categories and specific recommendations."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_completion(
            messages=messages,
            model=model,
            max_tokens=1024,  # Free tier limit
            temperature=0.7
        )
        
        if response.success:
            # Try to parse JSON from response
            try:
                json_start = response.content.find('{')
                json_end = response.content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    analysis_data = json.loads(response.content[json_start:json_end])
                    
                    return {
                        "success": True,
                        "provider": "groq",
                        "model_used": response.model_used,
                        "data": analysis_data,
                        "tokens_used": response.tokens_used,
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                else:
                    return {
                        "success": True,
                        "provider": "groq",
                        "model_used": response.model_used,
                        "data": {"analysis": response.content},
                        "tokens_used": response.tokens_used,
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "provider": "groq",
                    "model_used": response.model_used,
                    "data": {"analysis": response.content},
                    "tokens_used": response.tokens_used,
                    "response_time": response.response_time,
                    "raw_response": response.content
                }
        else:
            return {
                "success": False,
                "provider": "groq",
                "error": response.error,
                "model_attempted": model
            }

    def get_available_models(self) -> List[str]:
        """Get list of available free models"""
        return self.free_models.copy()

# Test the integration
if __name__ == "__main__":
    print("⚡ Testing Groq LLM Integration (Ultra-Fast Inference)")
    print("=" * 60)
    
    provider = GroqProvider()
    
    # Test connection
    if provider.test_connection():
        print("✅ Groq connection successful!")
        
        # Test speed comparison
        print("\n🏎️ Testing inference speed...")
        start_time = time.time()
        
        result = provider.market_analysis_request(
            "Analyze current trends in AI automation for small businesses. Provide 2 trends and 2 pain points.",
            model="llama3-8b-8192"  # Fastest model
        )
        
        if result.get('success'):
            print("✅ Market analysis successful!")
            print(f"📊 Model used: {result.get('model_used')}")
            print(f"📝 Response length: {len(result.get('raw_response', ''))}")
            print(f"🪙 Tokens used: {result.get('tokens_used')}")
            print(f"⚡ Response time: {result.get('response_time', 0):.2f}s")
            print(f"🚀 Speed advantage: Ultra-fast inference!")
        else:
            print(f"❌ Market analysis failed: {result.get('error')}")
    else:
        print("❌ Groq connection failed")
        print("Check your GROQ_API_KEY in the .env file")
