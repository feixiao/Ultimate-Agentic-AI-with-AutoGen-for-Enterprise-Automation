import os
import os
import sys

# Use pyautogen 0.7.5 API
import autogen

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration settings from model_config module
from model_config import llm_config

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
