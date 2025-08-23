import os
from dotenv import load_dotenv
from pathlib import Path

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

def get_secret(secret_name: str) -> str:
    """
    Retrieves a secret API key or configuration value from environment variables.

    Args:
        secret_name (str): The name of the environment variable to retrieve
                           (e.g., "SUPABASE_KEY").

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the specified secret is not found in the environment,
                    preventing the application from running with a missing key.
    """
    secret = os.getenv(secret_name)
    if not secret:
        print(f"❌ FATAL ERROR: Secret '{secret_name}' not found.")
        print("Please ensure it is defined in your .env file in the project root.")
        raise ValueError(f"API key '{secret_name}' not found in environment.")
    
    # Optional: Add a print statement for debugging to confirm which keys are being loaded.
    # Be careful with this in production to avoid leaking partial keys in logs.
    # print(f"✅ Loaded secret: {secret_name}")
    
    return secret

# --- Example Usage ---
# You can run this file directly to test if your .env keys are loading correctly.
if __name__ == '__main__':
    print("--- Running Secrets Manager Test ---")
    try:
        # Test a key you know is in your .env file
        supabase_url = get_secret("SUPABASE_URL")
        print(f"Successfully retrieved SUPABASE_URL: {supabase_url[:15]}...") # Print a snippet for verification

        # Test a key that you know is NOT in your .env file to see the error
        print("\nTesting for a missing key (this is expected to fail)...")
        get_secret("NON_EXISTENT_KEY")

    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n--- Test Complete ---")
