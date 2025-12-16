import os

# llm_config for local Ollama
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

llm_config = {
    "config_list": [
        {
            "model": ollama_model,
            "api_type": "ollama",
            "api_base": ollama_host,
        }
    ],
    # Disable disk caching to avoid sqlite schema issues
    "cache_seed": None,
    # Optional runtime params
    "temperature": 0,
    "timeout": 120,
}

__all__ = ["llm_config", "ollama_host", "ollama_model"]