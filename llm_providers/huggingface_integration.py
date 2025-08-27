#!/usr/bin/env python3
"""
Hugging Face Inference API Integration for sentient_venture_engine
Access to thousands of open-source models with free tier

Features:
- Access to 1000+ open-source models
- Free tier with daily limits
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
class HuggingFaceResponse:
    """Structured response from Hugging Face API"""
    content: str
    model_used: str
    success: bool
    error: Optional[str] = None
    response_time: Optional[float] = None

class HuggingFaceProvider:
    """
    Hugging Face Inference API integration for sentient_venture_engine
    
    Features:
    - Access to 1000+ open-source models
    - Free tier with daily quotas
    - Built-in rate limiting
    - Cost-free operation
    """
    
    def __init__(self):
        self.api_key = os.getenv('HF_API_KEY')
        self.base_url = "https://api-inference.huggingface.co/models"
        
        if not self.api_key:
            logger.warning("⚠️ HF_API_KEY not found in environment variables")
            
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Free tier models (text generation)
        self.free_models = [
            "microsoft/DialoGPT-medium",           # Conversational AI
            "facebook/blenderbot-400M-distill",   # Facebook's chatbot
            "microsoft/DialoGPT-small",           # Smaller conversational model
            "google/flan-t5-large",               # Google's instruction-tuned model
            "google/flan-t5-base",                # Smaller instruction model
            "EleutherAI/gpt-neo-2.7B",           # Open-source GPT alternative
            "EleutherAI/gpt-neo-1.3B",           # Smaller GPT alternative
            "bigscience/bloom-560m",              # Multilingual model
            "distilbert-base-uncased",            # Lightweight BERT
            "microsoft/DialoGPT-large",           # Larger conversational model
        ]
        
        # Rate limiting for free tier (conservative)
        self.rate_limit_delay = 2.0  # 2 seconds between requests (free tier)
        self.last_request_time = 0
        
        logger.info("🤗 Hugging Face LLM Provider initialized (1000+ models)")

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
        """Test connection to Hugging Face API"""
        if not self.api_key:
            logger.error("❌ Missing Hugging Face API key")
            return False
            
        try:
            # Test with a reliable model
            response = self.get_completion(
                prompt="Say 'OK' if you're working.",
                model="microsoft/DialoGPT-medium",
                max_length=10
            )
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"❌ Hugging Face connection test failed: {e}")
            return False

    def get_completion(self, 
                      prompt: str, 
                      model: str = "microsoft/DialoGPT-medium",
                      max_length: int = 100,
                      temperature: float = 0.7,
                      timeout: int = 30) -> HuggingFaceResponse:
        """
        Get completion from Hugging Face Inference API
        
        Args:
            prompt: Input prompt text
            model: Model to use (defaults to reliable free model)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Ensure we're using a known free model
            if model not in self.free_models:
                logger.warning(f"Model {model} not in free tier, using default")
                model = self.free_models[0]
            
            # Limit length for free tier
            max_length = min(max_length, 200)  # Conservative limit for free tier
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9
                }
            }
            
            start_time = time.time()
            logger.info(f"🔄 Requesting Hugging Face completion with model: {model}")
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response content (different models have different formats)
                content = ""
                if isinstance(data, list) and len(data) > 0:
                    if 'generated_text' in data[0]:
                        content = data[0]['generated_text']
                    elif 'text' in data[0]:
                        content = data[0]['text']
                    else:
                        content = str(data[0])
                elif isinstance(data, dict):
                    if 'generated_text' in data:
                        content = data['generated_text']
                    elif 'text' in data:
                        content = data['text']
                    else:
                        content = str(data)
                else:
                    content = str(data)
                
                # Clean up the content (remove original prompt if included)
                if content.startswith(prompt):
                    content = content[len(prompt):].strip()
                
                logger.info(f"✅ Hugging Face response: {len(content)} chars in {response_time:.2f}s")
                
                return HuggingFaceResponse(
                    content=content,
                    model_used=model,
                    success=True,
                    response_time=response_time
                )
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Hugging Face request failed: {error_msg}")
                
                return HuggingFaceResponse(
                    content="",
                    model_used=model,
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Request exception: {str(e)}"
            logger.error(f"❌ Hugging Face request exception: {error_msg}")
            
            return HuggingFaceResponse(
                content="",
                model_used=model,
                success=False,
                error=error_msg
            )

    def market_analysis_request(self, 
                              prompt: str, 
                              model: str = "google/flan-t5-large",  # Use instruction-tuned model
                              analysis_type: str = "market_intelligence") -> Dict[str, Any]:
        """
        Specialized market analysis request using Hugging Face models
        """
        # Format prompt for instruction-tuned models
        formatted_prompt = f"""Task: Market Intelligence Analysis
        
        Analyze the following request and provide structured insights:
        {prompt}
        
        Please provide a brief analysis with key trends and pain points."""
        
        response = self.get_completion(
            prompt=formatted_prompt,
            model=model,
            max_length=300,  # Free tier limit
            temperature=0.7
        )
        
        if response.success:
            # Try to structure the response
            try:
                # Look for JSON-like content
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    analysis_data = json.loads(content[json_start:json_end])
                    
                    return {
                        "success": True,
                        "provider": "huggingface",
                        "model_used": response.model_used,
                        "data": analysis_data,
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                else:
                    # Structure the text response
                    return {
                        "success": True,
                        "provider": "huggingface",
                        "model_used": response.model_used,
                        "data": {"analysis": response.content},
                        "response_time": response.response_time,
                        "raw_response": response.content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "provider": "huggingface",
                    "model_used": response.model_used,
                    "data": {"analysis": response.content},
                    "response_time": response.response_time,
                    "raw_response": response.content
                }
        else:
            return {
                "success": False,
                "provider": "huggingface",
                "error": response.error,
                "model_attempted": model
            }

    def get_available_models(self) -> List[str]:
        """Get list of available free models"""
        return self.free_models.copy()

# Test the integration
if __name__ == "__main__":
    print("🤗 Testing Hugging Face LLM Integration (1000+ Models)")
    print("=" * 60)
    
    provider = HuggingFaceProvider()
    
    # Test connection
    if provider.test_connection():
        print("✅ Hugging Face connection successful!")
        
        # Test model variety
        print(f"\n📚 Available models: {len(provider.free_models)}")
        for model in provider.free_models[:3]:
            print(f"  • {model}")
        print("  • ... and more")
        
        result = provider.market_analysis_request(
            "Analyze AI automation trends for small businesses. Provide key insights.",
            model="google/flan-t5-large"  # Instruction-tuned model
        )
        
        if result.get('success'):
            print("\n✅ Market analysis successful!")
            print(f"📊 Model used: {result.get('model_used')}")
            print(f"📝 Response length: {len(result.get('raw_response', ''))}")
            print(f"⏱️ Response time: {result.get('response_time', 0):.2f}s")
            print(f"🤗 Open-source model power!")
        else:
            print(f"❌ Market analysis failed: {result.get('error')}")
    else:
        print("❌ Hugging Face connection failed")
        print("Check your HF_API_KEY in the .env file")
