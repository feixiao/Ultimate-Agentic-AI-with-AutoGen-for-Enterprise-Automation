import os
from dotenv import load_dotenv
load_dotenv()


GENERATION_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
AGENT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
API_KEY = "<together_ai_key>"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# config keys - TogetherAI Llama
config_list = [
    {
        "model": AGENT_MODEL,
        "api_type":"together",
        "api_key": API_KEY,
        "stream":False
    }
]

# llm config
llm_config= {
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": None,
        "temperature": 0.7,
        "request_timeout": 120,
        "max_retries": 3,
        "seed": 42 # for reproducibility
    }

