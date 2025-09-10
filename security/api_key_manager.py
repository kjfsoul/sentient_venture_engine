import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_environment():
    """
    Finds and loads the .env file from the project root.

    This function intelligently searches for the .env file by traversing up
    from the current file's directory. This makes the key manager robust
    and callable from any script within the project structure without
    breaking the path to the .env file.
    """
    # Get the directory of the current script
    current_dir = Path(__file__).parent
    # Traverse up to find the project root (where .env should be)
    project_root = current_dir.parent
    dotenv_path = project_root / '.env'

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # Fallback for cases where the script is run from the root
        load_dotenv()

# Load the environment variables once when the module is imported
load_environment()

class KeyPriority(Enum):
    """Priority levels for API keys"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    OPTIONAL = "optional"

class APIKeyManager:
    """Enhanced API Key Manager with assessment and graceful degradation"""
    
    # Define key priorities and fallbacks
    KEY_CONFIG = {
        # Critical keys (system cannot function without these)
        "SUPABASE_URL": {"priority": KeyPriority.CRITICAL, "fallback": None},
        "SUPABASE_KEY": {"priority": KeyPriority.CRITICAL, "fallback": None},
        
        # High priority keys (major functionality impact)
        "OPENROUTER_API_KEY": {"priority": KeyPriority.HIGH, "fallback": ["OPENAI_API_KEY", "GROQ_API_KEY"]},
        "GITHUB_TOKEN": {"priority": KeyPriority.HIGH, "fallback": None},
        
        # Medium priority keys (good alternatives exist)
        "OPENAI_API_KEY": {"priority": KeyPriority.MEDIUM, "fallback": ["GROQ_API_KEY", "TOGETHER_API_KEY"]},
        "SERPER_API_KEY": {"priority": KeyPriority.MEDIUM, "fallback": None},
        "GROQ_API_KEY": {"priority": KeyPriority.MEDIUM, "fallback": ["TOGETHER_API_KEY"]},
        "TOGETHER_API_KEY": {"priority": KeyPriority.MEDIUM, "fallback": None},
        "GEMINI_API_KEY": {"priority": KeyPriority.MEDIUM, "fallback": None},
        
        # Optional keys (nice to have)
        "HF_API_KEY": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "ROUTELLM_API_KEY": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "PEXELS_API_KEY": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "DEEPAI_API_KEY": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "REDDIT_CLIENT_ID": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "REDDIT_CLIENT_SECRET": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "NOTION_API_KEY": {"priority": KeyPriority.OPTIONAL, "fallback": None},
        "VERCEL_TOKEN": {"priority": KeyPriority.OPTIONAL, "fallback": None},
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._missing_keys = set()
        self._available_keys = set()
        self._assess_keys()
    
    def _assess_keys(self):
        """Assess availability of all configured keys"""
        for key_name, config in self.KEY_CONFIG.items():
            if self.is_key_available(key_name, silent=True):
                self._available_keys.add(key_name)
            else:
                self._missing_keys.add(key_name)
    
    def is_key_available(self, key_name: str, silent: bool = False) -> bool:
        """Check if an API key is available and valid"""
        value = os.getenv(key_name)
        
        # Check for missing or placeholder values
        invalid_values = {
            None, "", 
            "your-id", "your-secret", "your-key", "your-token",
            f"your_{key_name.lower()}", 
            f"your-{key_name.lower().replace('_', '-')}"
        }
        
        is_available = value not in invalid_values
        
        if not silent and not is_available:
            self.logger.warning(f"⚠️ API key '{key_name}' is missing or invalid")
        
        return is_available
    
    def get_secret_with_fallback(self, primary_key: str, fallback_keys: List[str] = None, required: bool = True) -> Optional[str]:
        """Get secret with automatic fallback to alternative keys"""
        # Try primary key first
        if self.is_key_available(primary_key, silent=True):
            return os.getenv(primary_key)
        
        # Try configured fallbacks
        config = self.KEY_CONFIG.get(primary_key, {})
        configured_fallbacks = config.get("fallback", [])
        
        # Combine provided and configured fallbacks
        all_fallbacks = (fallback_keys or []) + (configured_fallbacks or [])
        
        for fallback_key in all_fallbacks:
            if self.is_key_available(fallback_key, silent=True):
                self.logger.info(f"🔄 Using fallback '{fallback_key}' for '{primary_key}'")
                return os.getenv(fallback_key)
        
        # Handle missing required keys
        if required:
            priority = self.KEY_CONFIG.get(primary_key, {}).get("priority", KeyPriority.MEDIUM)
            
            if priority == KeyPriority.CRITICAL:
                error_msg = f"CRITICAL: API key '{primary_key}' is required for core functionality"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            elif priority == KeyPriority.HIGH:
                self.logger.error(f"❌ HIGH PRIORITY: API key '{primary_key}' missing - functionality will be limited")
            else:
                self.logger.warning(f"⚠️ Optional API key '{primary_key}' missing - some features unavailable")
        
        return None
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on API keys"""
        providers = []
        
        llm_keys = {
            "OPENROUTER_API_KEY": "openrouter",
            "OPENAI_API_KEY": "openai", 
            "GROQ_API_KEY": "groq",
            "TOGETHER_API_KEY": "together",
            "GEMINI_API_KEY": "gemini",
            "HF_API_KEY": "huggingface"
        }
        
        for key, provider in llm_keys.items():
            if self.is_key_available(key, silent=True):
                providers.append(provider)
        
        return providers
    
    def get_system_capabilities(self) -> Dict[str, bool]:
        """Assess what system capabilities are available based on API keys"""
        capabilities = {
            "llm_analysis": bool(self.get_available_llm_providers()),
            "database_storage": self.is_key_available("SUPABASE_KEY", silent=True),
            "code_analysis": self.is_key_available("GITHUB_TOKEN", silent=True),
            "web_search": self.is_key_available("SERPER_API_KEY", silent=True),
            "image_processing": any([
                self.is_key_available("PEXELS_API_KEY", silent=True),
                self.is_key_available("DEEPAI_API_KEY", silent=True)
            ]),
            "social_analysis": all([
                self.is_key_available("REDDIT_CLIENT_ID", silent=True),
                self.is_key_available("REDDIT_CLIENT_SECRET", silent=True)
            ])
        }
        
        return capabilities
    
    def get_missing_critical_keys(self) -> List[str]:
        """Get list of missing critical API keys"""
        missing_critical = []
        
        for key_name, config in self.KEY_CONFIG.items():
            if config["priority"] == KeyPriority.CRITICAL and not self.is_key_available(key_name, silent=True):
                missing_critical.append(key_name)
        
        return missing_critical
    
    def print_status_report(self):
        """Print a comprehensive status report"""
        print("\n🔐 API Key Status Report")
        print("=" * 50)
        
        # System capabilities
        capabilities = self.get_system_capabilities()
        print("\n📊 System Capabilities:")
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"  {status} {capability.replace('_', ' ').title()}")
        
        # LLM providers
        llm_providers = self.get_available_llm_providers()
        print(f"\n🧠 Available LLM Providers ({len(llm_providers)}):")
        for provider in llm_providers:
            print(f"  ✅ {provider}")
        
        # Missing critical keys
        missing_critical = self.get_missing_critical_keys()
        if missing_critical:
            print(f"\n🚨 Missing Critical Keys:")
            for key in missing_critical:
                print(f"  ❌ {key}")
        
        # Summary by priority
        print("\n📋 Key Status by Priority:")
        for priority in KeyPriority:
            keys_for_priority = [
                key for key, config in self.KEY_CONFIG.items() 
                if config["priority"] == priority
            ]
            
            available_count = sum(
                1 for key in keys_for_priority 
                if self.is_key_available(key, silent=True)
            )
            
            total_count = len(keys_for_priority)
            percentage = (available_count / total_count * 100) if total_count > 0 else 0
            
            print(f"  {priority.value.title()}: {available_count}/{total_count} ({percentage:.0f}%)")

# Global instance for backward compatibility
_key_manager = APIKeyManager()

def get_secret(secret_name: str) -> str:
    """
    Retrieves a secret API key or configuration value from environment variables.
    Maintains backward compatibility while adding enhanced error handling.

    Args:
        secret_name (str): The name of the environment variable to retrieve

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the specified secret is not found in the environment
    """
    secret = os.getenv(secret_name)
    if not secret or secret in ["your-id", "your-secret", "your-key", "your-token", f"your_{secret_name.lower()}"]:
        print(f"❌ FATAL ERROR: Secret '{secret_name}' not found or invalid.")
        print("Please ensure it is defined in your .env file in the project root.")
        raise ValueError(f"API key '{secret_name}' not found in environment.")
    
    return secret

def get_secret_optional(secret_name: str, fallback_keys: List[str] = None) -> Optional[str]:
    """
    Get an optional secret with fallback support.
    
    Args:
        secret_name: Primary key to retrieve
        fallback_keys: List of fallback keys to try
    
    Returns:
        The secret value or None if not available
    """
    return _key_manager.get_secret_with_fallback(secret_name, fallback_keys, required=False)

def get_available_llm_provider() -> Optional[str]:
    """
    Get the first available LLM provider for backward compatibility.
    
    Returns:
        Name of available provider or None
    """
    providers = _key_manager.get_available_llm_providers()
    return providers[0] if providers else None

def check_system_readiness() -> bool:
    """
    Check if the system has minimum required API keys to function.
    
    Returns:
        True if system can function, False if critical keys are missing
    """
    missing_critical = _key_manager.get_missing_critical_keys()
    return len(missing_critical) == 0

# --- Example Usage and Testing ---
if __name__ == '__main__':
    print("🔐 Running Enhanced API Key Manager Test")
    print("=" * 60)
    
    # Test the enhanced key manager
    key_manager = APIKeyManager()
    
    # Print comprehensive status report
    key_manager.print_status_report()
    
    # Test system readiness
    print(f"\n📈 System Readiness Check")
    is_ready = check_system_readiness()
    print(f"System Ready: {'✅ Yes' if is_ready else '❌ No'}")
    
    # Test LLM provider availability
    print(f"\n🧠 LLM Provider Test")
    available_provider = get_available_llm_provider()
    print(f"Primary LLM Provider: {available_provider or 'None available'}")
    
    # Test optional key retrieval
    print(f"\n🔑 Optional Key Test")
    try:
        # Test getting an optional key with fallbacks
        optional_key = get_secret_optional("SOME_OPTIONAL_KEY", ["OPENROUTER_API_KEY", "OPENAI_API_KEY"])
        print(f"Optional key result: {'Found fallback' if optional_key else 'Not available'}")
    except Exception as e:
        print(f"Error testing optional key: {e}")
    
    # Test backward compatibility
    print(f"\n🔄 Backward Compatibility Test")
    try:
        # Test a key that should exist
        test_key = "SUPABASE_URL"
        if _key_manager.is_key_available(test_key, silent=True):
            supabase_url = get_secret(test_key)
            print(f"✅ Successfully retrieved {test_key}: {supabase_url[:20]}...")
        else:
            print(f"⚠️ {test_key} not available for testing")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n🎉 Test Complete")
