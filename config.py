import os
from dotenv import load_dotenv
# request retries
import requests
from requests.adapters import HTTPAdapter 
from urllib3.util.retry import Retry

# Set up the environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = os.getenv("SERPAPI_URL", "https://serpapi.com/search")
SEARCH_ENGINE = "google"
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

# Prohibit TensorFlow oneDNN backend to avoid envrionment issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Proxy
# os.environ['NO_PROXY'] = '127.0.0.1,localhost'

# Request timeout and retries
requests.adapters.DEFAULT_RETRIES = 3