# sentient_venture_engine/test_api.py

import sys
from langchain_openai import ChatOpenAI

# This script tests the most critical part: loading the API key
# and initializing the connection to the language model.

try:
    # Attempt to import the secret manager
    from security.api_key_manager import get_secret
    print("✅ Successfully imported the secret manager.")
except ImportError:
    print("❌ FATAL: Could not import 'get_secret'.")
    print("   Ensure 'security/api_key_manager.py' exists and is in the correct folder.")
    sys.exit(1)

try:
    # Attempt to load the API key from your .env file
    api_key = get_secret("OPENROUTER_API_KEY")
    print(f"✅ Successfully loaded OPENROUTER_API_KEY: sk-or-v1...{api_key[-4:]}")
except ValueError as e:
    print("❌ FATAL: Failed to load the API key from your .env file.")
    print(f"   Error: {e}")
    print("   Please check that your .env file exists in the project root and the key name is correct.")
    sys.exit(1)

try:
    # Attempt to initialize the LLM. This will fail if the key is invalid.
    print("\n⏳ Initializing LLM with OpenRouter...")
    llm = ChatOpenAI(
        model="anthropic/claude-3-opus",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"HTTP-Referer": "https://sve.ai", "X-Title": "SVE"}
    )
    print("✅ LLM Initialized Successfully.")
    
    # Final test: Make a simple API call
    print("⏳ Making a test API call...")
    response = llm.invoke("Test prompt")
    print("✅ Test API call successful!")
    print("\n🎉 SUCCESS: Your .env file and API key are working correctly.")

except Exception as e:
    print("\n❌ FATAL: LLM initialization or API call failed.")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Error Details: {e}")
    print("\n   This almost always means the API key in your .env file is invalid or lacks funds.")
    print("   Please verify the key on the OpenRouter website.")
    sys.exit(1)
