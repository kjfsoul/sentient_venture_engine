#!/usr/bin/env python3
"""
Together.ai LLM Integration for sentient_venture_engine
Fourth LLM provider option specializing in open-source models

Together.ai provides access to high-quality open-source LLMs with competitive pricing
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
class TogetherResponse:
    """Structured response from Together.ai API"""
    content: str
    model_used: str
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None

class TogetherAIProvider:
    """
    Together.ai integration for sentient_venture_engine
    
    Features:
    - High-quality open-source models
    - Competitive pricing
    - Fast inference
    - Multiple model options
    """
    
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        self.base_url = "https://api.together.xyz/v1"
        
        if not self.api_key:
            logger.warning("⚠️ TOGETHER_API_KEY not found in environment variables")
            
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Popular Together.ai models
        self.available_models = [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf", 
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "Open-Orca/Mistral-7B-OpenOrca"
        ]
        
        logger.info("🤝 Together.ai LLM Provider initialized")

    def test_connection(self) -> bool:
        """Test connection to Together.ai API"""
        if not self.api_key:
            logger.error("❌ Missing Together.ai API key")
            return False
            
        try:
            # Test with a simple request using a reliable model
            response = self.get_completion(
                messages=[{"role": "user", "content": "Say 'OK' if you're working."}],
                model="mistralai/Mistral-7B-Instruct-v0.1",
                max_tokens=10
            )
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"❌ Together.ai connection test failed: {e}")
            return False

    def get_completion(self, 
                      messages: List[Dict[str, str]], 
                      model: str = "mistralai/Mistral-7B-Instruct-v0.1",
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      timeout: int = 30) -> TogetherResponse:
        """
        Get completion from Together.ai API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "stop": None
            }
            
            logger.info(f"🔄 Requesting Together.ai completion with model: {model}")
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response content
                content = ""
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0].get('message', {}).get('content', '')
                
                # Extract usage info
                usage = data.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
                
                logger.info(f"✅ Together.ai response: {len(content)} chars, {tokens_used} tokens")
                
                return TogetherResponse(
                    content=content,
                    model_used=model,
                    success=True,
                    tokens_used=tokens_used
                )
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Together.ai request failed: {error_msg}")
                
                return TogetherResponse(
                    content="",
                    model_used=model,
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Request exception: {str(e)}"
            logger.error(f"❌ Together.ai request exception: {error_msg}")
            
            return TogetherResponse(
                content="",
                model_used=model,
                success=False,
                error=error_msg
            )

    def market_analysis_request(self, 
                              prompt: str, 
                              model: str = "meta-llama/Llama-2-70b-chat-hf",
                              analysis_type: str = "market_intelligence") -> Dict[str, Any]:
        """
        Specialized market analysis request using Together.ai
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
            max_tokens=2000,
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
                        "provider": "together",
                        "model_used": response.model_used,
                        "data": analysis_data,
                        "tokens_used": response.tokens_used,
                        "raw_response": response.content
                    }
                else:
                    return {
                        "success": True,
                        "provider": "together",
                        "model_used": response.model_used,
                        "data": {"analysis": response.content},
                        "tokens_used": response.tokens_used,
                        "raw_response": response.content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "provider": "together",
                    "model_used": response.model_used,
                    "data": {"analysis": response.content},
                    "tokens_used": response.tokens_used,
                    "raw_response": response.content
                }
        else:
            return {
                "success": False,
                "provider": "together",
                "error": response.error,
                "model_attempted": model
            }

    def get_available_models(self) -> List[str]:
        """Get list of available models from Together.ai"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                return models if models else self.available_models
            else:
                logger.warning("Could not fetch models list, using defaults")
                return self.available_models
        except Exception as e:
            logger.warning(f"Error fetching models: {e}, using defaults")
            return self.available_models

# Test the integration
if __name__ == "__main__":
    print("🤝 Testing Together.ai LLM Integration")
    print("=" * 50)
    
    provider = TogetherAIProvider()
    
    # Test connection
    if provider.test_connection():
        print("✅ Together.ai connection successful!")
        
        # Test market analysis
        result = provider.market_analysis_request(
            "Analyze current trends in AI automation for small businesses. Provide 2 trends and 2 pain points.",
            model="mistralai/Mistral-7B-Instruct-v0.1"  # Use faster model for testing
        )
        
        if result.get('success'):
            print("✅ Market analysis successful!")
            print(f"📊 Model used: {result.get('model_used')}")
            print(f"📝 Response length: {len(result.get('raw_response', ''))}")
            print(f"🪙 Tokens used: {result.get('tokens_used')}")
        else:
            print(f"❌ Market analysis failed: {result.get('error')}")
    else:
        print("❌ Together.ai connection failed")
        print("Check your TOGETHER_API_KEY in the .env file")
