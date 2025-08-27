#!/usr/bin/env python3
"""
Abacus.ai LLM Teams Integration for sentient_venture_engine
Alternative LLM provider with potentially better cost structure and availability

Note: Requires deployed LLM model on Abacus.ai platform
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
class AbacusLLMResponse:
    """Structured response from Abacus.ai LLM Teams"""
    content: str
    deployment_id: str
    conversation_id: str
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None

class AbacusLLMTeams:
    """
    Abacus.ai LLM Teams integration for sentient_venture_engine
    
    Features:
    - Chat completion API
    - Conversation management
    - Cost tracking
    - Fallback strategies
    """
    
    def __init__(self):
        self.api_key = os.getenv('ABACUS_API_KEY')
        self.deployment_id = os.getenv('ABACUS_DEPLOYMENT_ID')
        self.deployment_token = os.getenv('ABACUS_DEPLOYMENT_TOKEN')
        self.base_url = "https://api.abacus.ai/api/v0"
        
        if not self.api_key:
            logger.warning("⚠️ ABACUS_API_KEY not found in environment variables")
        if not self.deployment_id:
            logger.warning("⚠️ ABACUS_DEPLOYMENT_ID not found in environment variables")
        if not self.deployment_token:
            logger.warning("⚠️ ABACUS_DEPLOYMENT_TOKEN not found in environment variables")
            
        self.session = requests.Session()
        self.session.headers.update({
            'apiKey': self.api_key,  # Abacus uses 'apiKey' header
            'Content-Type': 'application/json'
        })
        
        logger.info("🤖 Abacus.ai LLM Teams client initialized")

    def test_connection(self) -> bool:
        """Test connection to Abacus.ai API"""
        if not all([self.api_key, self.deployment_id, self.deployment_token]):
            logger.error("❌ Missing required Abacus.ai credentials")
            return False
            
        try:
            # Test with a simple request
            response = self.get_chat_response(
                messages=[{"is_user": True, "text": "Say 'OK' if you're working."}],
                max_tokens=10
            )
            return response.success and "OK" in response.content.upper()
        except Exception as e:
            logger.error(f"❌ Abacus.ai connection test failed: {e}")
            return False

    def get_chat_response(self, 
                         messages: List[Dict[str, Any]], 
                         llm_name: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         system_message: Optional[str] = None,
                         timeout: int = 30) -> AbacusLLMResponse:
        """
        Get chat response from Abacus.ai deployed LLM
        
        Args:
            messages: List of message dicts with 'is_user' (bool) and 'text' (str)
            llm_name: Specific LLM backend to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_message: System prompt
            timeout: Request timeout in seconds
        """
        try:
            payload = {
                "deploymentId": self.deployment_id,
                "deploymentToken": self.deployment_token,
                "messages": messages,
                "numCompletionTokens": max_tokens,
                "temperature": temperature
            }
            
            if llm_name:
                payload["llmName"] = llm_name
            if system_message:
                payload["systemMessage"] = system_message
            
            logger.info(f"🔄 Requesting Abacus.ai chat completion...")
            
            response = self.session.post(
                f"{self.base_url}/getChatResponse",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    # Extract the latest message (bot response)
                    messages_list = data.get('messages', [])
                    if messages_list:
                        latest_message = messages_list[-1]
                        content = latest_message.get('text', '')
                    else:
                        content = "No response generated"
                    
                    conversation_id = data.get('deploymentConversationId', '')
                    
                    logger.info(f"✅ Abacus.ai response: {len(content)} chars")
                    
                    return AbacusLLMResponse(
                        content=content,
                        deployment_id=self.deployment_id,
                        conversation_id=conversation_id,
                        success=True
                    )
                else:
                    error_msg = data.get('error', 'Unknown error')
                    logger.error(f"❌ Abacus.ai API error: {error_msg}")
                    
                    return AbacusLLMResponse(
                        content="",
                        deployment_id=self.deployment_id,
                        conversation_id="",
                        success=False,
                        error=error_msg
                    )
                    
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Abacus.ai request failed: {error_msg}")
                
                return AbacusLLMResponse(
                    content="",
                    deployment_id=self.deployment_id,
                    conversation_id="",
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Request exception: {str(e)}"
            logger.error(f"❌ Abacus.ai request exception: {error_msg}")
            
            return AbacusLLMResponse(
                content="",
                deployment_id=self.deployment_id,
                conversation_id="",
                success=False,
                error=error_msg
            )

    def market_analysis_request(self, 
                              prompt: str, 
                              analysis_type: str = "market_intelligence") -> Dict[str, Any]:
        """
        Specialized market analysis request for sentient_venture_engine
        """
        system_message = """You are a senior market intelligence analyst specializing in venture strategy and business opportunity identification. 
        Provide structured, actionable insights based on current market trends, competitive landscapes, and emerging opportunities.
        Format your response as valid JSON with clear categories and specific recommendations."""
        
        messages = [
            {"is_user": True, "text": prompt}
        ]
        
        response = self.get_chat_response(
            messages=messages,
            max_tokens=1500,
            temperature=0.3,  # Lower temperature for more focused analysis
            system_message=system_message
        )
        
        if response.success:
            try:
                # Try to parse as JSON
                json_start = response.content.find('{')
                json_end = response.content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    analysis_data = json.loads(response.content[json_start:json_end])
                    
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "deployment_id": response.deployment_id,
                        "conversation_id": response.conversation_id,
                        "data": analysis_data,
                        "raw_response": response.content
                    }
                else:
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "deployment_id": response.deployment_id,
                        "conversation_id": response.conversation_id,
                        "data": {"analysis": response.content},
                        "raw_response": response.content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "deployment_id": response.deployment_id,
                    "conversation_id": response.conversation_id,
                    "data": {"analysis": response.content},
                    "raw_response": response.content
                }
        else:
            return {
                "success": False,
                "analysis_type": analysis_type,
                "error": response.error,
                "deployment_id": response.deployment_id
            }

def test_abacus_integration():
    """Test function for Abacus.ai integration"""
    print("🚀 Testing Abacus.ai LLM Teams integration...")
    
    client = AbacusLLMTeams()
    
    # Test 1: Basic connection
    print("\n1. Testing basic connection...")
    if client.test_connection():
        print("✅ Abacus.ai connection: SUCCESS")
    else:
        print("❌ Abacus.ai connection: FAILED")
        return False
    
    # Test 2: Market analysis request
    print("\n2. Testing market analysis request...")
    analysis_result = client.market_analysis_request(
        prompt="""Analyze the current AI automation market for small businesses. 
        Identify 3 key trends and 3 pain points. Return as JSON with 'trends' and 'pain_points' arrays.""",
        analysis_type="ai_automation_market"
    )
    
    if analysis_result.get("success"):
        print("✅ Market analysis: SUCCESS")
        print(f"📊 Model used: {analysis_result.get('model_used')}")
        print(f"🎯 Tokens used: {analysis_result.get('tokens_used')}")
        print(f"💰 Cost: ${analysis_result.get('cost', 0):.4f}")
        print(f"📋 Analysis preview: {str(analysis_result.get('data', {}))[:200]}...")
    else:
        print("❌ Market analysis: FAILED")
        print(f"Error: {analysis_result.get('error')}")
    
    return analysis_result.get("success", False)

if __name__ == "__main__":
    test_abacus_integration()
