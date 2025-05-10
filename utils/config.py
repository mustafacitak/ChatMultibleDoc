import os
from dotenv import load_dotenv
import logging

# Load .env file
load_dotenv()

# Static configuration values
MODEL_NAME = "gemini-2.0-flash"
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "default_collection")
DB_PATH = "./db"
CONFIG_PATH = "./config"

# Get API key from .env file
def get_api_key():
    """
    Reads and returns the Google API key from .env file.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("GOOGLE_API_KEY not found. Make sure it's defined in your .env file.")
    return api_key

# Application initialization checks
def init_app():
    """
    Creates necessary directories and performs basic checks at application startup.
    """
    # Create required directories
    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(CONFIG_PATH, exist_ok=True)
    
    # Check API key
    api_key = get_api_key()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        logging.error("API key not found. Application may not work properly.")
    
    return api_key is not None 