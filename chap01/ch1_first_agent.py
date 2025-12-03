import os
import sys

# Use pyautogen 0.7.5 classic API
import autogen

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

# Create an assistant agent with a given name, LLM configuration, and system prompt.
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant.",
)

# Create a user proxy agent (0.7.5 supports human_input_mode)
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=1,
    code_execution_config={
        "use_docker": False,
        "last_n_messages": 3,
        "work_dir": "workspace",
    },
)

# Start a conversation by initiating a chat between the user proxy and the assistant agent
user_proxy.initiate_chat(assistant, message="Tell me a joke.")
