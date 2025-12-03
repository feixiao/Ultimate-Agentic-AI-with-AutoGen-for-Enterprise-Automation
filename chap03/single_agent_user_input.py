import autogen
import json
import os
import sys

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import configuration settings from the chap03 package for direct script execution
from chap03 import llm_config

# Define a prompt for the assistant agent to classify user queries
agent_prompt = """You are a helpful assistant that classifies and resolves user queries.
Query Classes:
1. Technical Support
2. Account Management
3. Billing Issues
4. Product Information
5. General Inquiry

For each query:
1. Understand the user's request
2. Classify it into one of the above categories
3. Provide an appropriate resolution
4. Format response as: 
   Category: [class]
   Resolution: [detailed response]
"""

# Create a user proxy agent for interfacing with the user
user = autogen.UserProxyAgent(
    name="user_proxy",  # Identifier for the user proxy
    human_input_mode="TERMINATE",  # Wait for termination signal for input
    max_consecutive_auto_reply=1,  # Limit consecutive auto replies
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # Disable code execution configuration
)

# Create an assistant agent for classifying queries using the prompt
assistant = autogen.AssistantAgent(
    name="query_classifier",  # Name of the assistant agent
    system_message=agent_prompt,  # Instruction prompt for the assistant
    llm_config=llm_config,  # Language model configuration
)

# List to store session history (each entry is a query-response pair)
session_history = []


def process_query(user_input):
    """
    Process a user query using the AutoGen agent.

    Parameters:
    - user_input: str, the query input from the user

    Returns:
    - str: Response from the assistant agent
    """
    # Initiate the chat with the assistant agent using the user's query
    user.initiate_chat(assistant, message=user_input)

    # Retrieve the chat history for the assistant agent
    chat_history = user.chat_messages[assistant]
    if chat_history:
        # Return the content of the last message from the assistant
        return chat_history[-1]["content"]
    # Return default message if no response is generated
    return "No response generated"


def main():
    """
    Main function to continuously accept user queries and process them.
    """
    while True:
        # Prompt the user for input
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        # Process the user's query using the assistant agent
        response = process_query(user_input)
        # Append the query and corresponding response to session history
        session_history.append({"query": user_input, "response": response})
        print("\nResponse:", response)


if __name__ == "__main__":
    main()
