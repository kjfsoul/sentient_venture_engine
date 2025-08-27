#!/usr/bin/env python3
"""
Google Gemini API Integration for sentient_venture_engine
Direct integration with Google AI Studio's Gemini models

Features:
- Direct Gemini API access (no OpenRouter)
- Free tier with monthly limits
- Built-in rate limiting
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
class GeminiResponse:
    """Structured response from Gemini API"""
    content: str
    model_used: str
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None

class GeminiProvider:
    """
    Google Gemini API integration for sentient_venture_engine
    
    Features:
    - Direct Gemini API access
    - Free tier with generous limits
    - Built-in safety features
    - Cost-free operation
    """
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
        
        # Available free models
        self.free_models = [
            "gemini-1.5-flash",        # Fast, efficient model
            "gemini-1.5-pro",          # More capable model
            "gemini-pro",              # Standard model
        ]
        
        # Rate limiting for free tier (generous limits)
        self.rate_limit_delay = 0.5  # 0.5 seconds between requests
        self.last_request_time = 0
        
        logger.info("💎 Gemini LLM Provider initialized (Google AI)")

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
        """Test connection to Gemini API"""
        if not self.api_key:
            logger.error("❌ Missing Gemini API key")
            return False
            
        try:
            # Test with fastest model
            response = self.get_completion(
                messages=[{"role": "user", "content": "Say 'OK' if you're working."}],
                model="gemini-1.5-flash",
                max_tokens=10
            )
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"❌ Gemini connection test failed: {e}")
            return False

    def get_completion(self, 
                      messages: List[Dict[str, str]], 
                      model: str = "gemini-1.5-flash",
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      timeout: int = 30) -> GeminiResponse:
        """
        Get completion from Gemini API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to fastest free model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            timeout: Request timeout in seconds
        """
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Ensure we're using a free model
            if model not in self.free_models:
                logger.warning(f"Model {model} not recognized, using default")
                model = self.free_models[0]
            
            # Convert messages to Gemini format
            gemini_parts = []
            for message in messages:
                if message.get("role") == "system":
                    # Gemini doesn't have system role, prepend to user message
                    continue
                elif message.get("role") == "user":
                    gemini_parts.append({"text": message.get("content", "")})
                elif message.get("role") == "assistant":
                    # For conversation history (not used in single request)
                    continue
            
            # If there was a system message, prepend it
            system_content = ""
            for message in messages:
                if message.get("role") == "system":
                    system_content = message.get("content", "")
                    break
            
            if system_content and gemini_parts:
                gemini_parts[0]["text"] = f"{system_content}\n\n{gemini_parts[0]['text']}"
            
            payload = {
                "contents": [{
                    "parts": gemini_parts
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": min(max_tokens, 2048),  # Free tier limit
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            start_time = time.time()
            logger.info(f"🔄 Requesting Gemini completion with model: {model}")
            
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response content
                content = ""
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            content = parts[0]['text']
                
                # Extract usage info if available
                usage_metadata = data.get('usageMetadata', {})
                tokens_used = usage_metadata.get('totalTokenCount', 0)
                
                logger.info(f"✅ Gemini response: {len(content)} chars, {tokens_used} tokens in {response_time:.2f}s")
                
                return GeminiResponse(
                    content=content,
                    model_used=model,
                    success=True,
                    tokens_used=tokens_used,
                    response_time=response_time
                )
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Gemini request failed: {error_msg}")
                
                return GeminiResponse(
                    content="",
                    model_used=model,
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Request exception: {str(e)}"
            logger.error(f"❌ Gemini request exception: {error_msg}")
            
            return GeminiResponse(
                content="",
                model_used=model,
                success=False,
                error=error_msg
            )

    def market_analysis_request(self, 
                              prompt: str, 
                              model: str = "gemini-1.5-pro",  # Use more capable model for analysis
                              analysis_type: str = "market_intelligence") -> Dict[str, Any]:
        """
        Specialized market analysis request using Gemini
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
            max_tokens=1500,  # Free tier allows generous limits
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
                        "provider": "gemini",
                        "model_used": response.model_used,
                        "data": analysis_data,
                        "tokens_used": response.tokens_used,
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                else:
                    return {
                        "success": True,
                        "provider": "gemini",
                        "model_used": response.model_used,
                        "data": {"analysis": response.content},
                        "tokens_used": response.tokens_used,
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "provider": "gemini",
                    "model_used": response.model_used,
                    "data": {"analysis": response.content},
                    "tokens_used": response.tokens_used,
                    "response_time": response.response_time,
                    "raw_response": response.content
                }
        else:
            return {
                "success": False,
                "provider": "gemini",
                "error": response.error,
                "model_attempted": model
            }

    def get_available_models(self) -> List[str]:
        """Get list of available free models"""
        return self.free_models.copy()

# Test the integration
if __name__ == "__main__":
    print("💎 Testing Gemini LLM Integration (Google AI)")
    print("=" * 50)
    
    provider = GeminiProvider()
    
    # Test connection
    if provider.test_connection():
        print("✅ Gemini connection successful!")
        
        # Test analysis capability
        result = provider.market_analysis_request(
            "Analyze current trends in AI automation for small businesses. Provide 2 trends and 2 pain points.",
            model="gemini-1.5-pro"
        )
        
        if result.get('success'):
            print("✅ Market analysis successful!")
            print(f"📊 Model used: {result.get('model_used')}")
            print(f"📝 Response length: {len(result.get('raw_response', ''))}")
            print(f"🪙 Tokens used: {result.get('tokens_used')}")
            print(f"⏱️ Response time: {result.get('response_time', 0):.2f}s")
            print(f"💎 Google AI quality!")
        else:
            print(f"❌ Market analysis failed: {result.get('error')}")
    else:
        print("❌ Gemini connection failed")
        print("Check your GEMINI_API_KEY in the .env file")
