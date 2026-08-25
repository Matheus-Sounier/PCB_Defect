import os

from dotenv import load_dotenv

load_dotenv()

def get_env_or_default(name: str, default: str = "") -> str:
    return os.getenv(name, default)
