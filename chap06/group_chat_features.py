"""
Module: Group Chat Features
This module demonstrates various features for managing group chats
with agent-based interactions, message formatting, and speaker selection.
"""

from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    register_function,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

# Configuration for language model (LLM)
llm_config = {}
agents = []

# -----------------------------------------------------------------------------
# Speaker Selection Methods
# -----------------------------------------------------------------------------
# Create a group chat instance with automatic speaker selection.
group_chat = GroupChat(
    speaker_selection_method="auto",
    selector_config={
        "model": "gpt-3.5-turbo",  # Specify model for speaker selection
        "temperature": 0.7,  # Control randomness in responses
    },
)

# -----------------------------------------------------------------------------
# Conversation Control
# -----------------------------------------------------------------------------
# Initialize a group chat with limited rounds and a starting system message.
group_chat = GroupChat(
    max_round=5,  # Maximum conversation rounds
    allow_repeat_speaker=False,  # Prevent same speaker consecutively
    messages=[{"role": "system", "content": "Initial context"}],  # Starting context
)

# -----------------------------------------------------------------------------
# Dynamic Interactions
# -----------------------------------------------------------------------------
# Create a user proxy agent for dynamic code execution interactions.
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding"
    },  # Working directory for code execution
)
# Add the user proxy to the group chat.
group_chat.add_agent(user_proxy)

# -----------------------------------------------------------------------------
# Message Formatting Example
# -----------------------------------------------------------------------------
# Define a sample message format.
message = {
    "role": "assistant",
    "content": "Message content",
    "name": "AgentName",
    "timestamp": "2024-02-16T12:00:00",  # ISO format timestamp
}

# -----------------------------------------------------------------------------
# Error Handling Configuration
# -----------------------------------------------------------------------------
# Create a group chat instance with error handling and timeout settings.
group_chat = GroupChat(
    agents=agents,  # List of agents participating
    max_round=10,  # Increase maximum conversation rounds
    timeout=300,  # Timeout in seconds (5 minutes)
)

# -----------------------------------------------------------------------------
# Customization Options: Termination Message
# -----------------------------------------------------------------------------


def custom_termination_msg(message):
    """
    Custom function to determine if the conversation should terminate.

    Args:
        message: The message object containing message details.

    Returns:
        bool: True if 'TERMINATE' is found in the message content.
    """
    return "TERMINATE" in message.content


# Create a group chat instance with a custom termination message function.
group_chat = GroupChat(
    agents=agents,
    termination_msg=custom_termination_msg,  # Function to check termination condition
)

# -----------------------------------------------------------------------------
# Integration with Group Chat Manager
# -----------------------------------------------------------------------------
# Instantiate a manager to coordinate group chat interactions.
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    system_message="Coordinate the group discussion",  # Instruction for coordination
)

# -----------------------------------------------------------------------------
# Custom Speaker Selection Logic
# -----------------------------------------------------------------------------
# Initialize agent placeholders for custom speaker selection.
user: UserProxyAgent = None
agent_a: AssistantAgent = None
agent_b: AssistantAgent = None
agent_c: AssistantAgent = None


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """
    Custom function to select the next speaker based on the last speaker.

    Args:
        last_speaker: The agent who spoke last.
        groupchat: The current group chat instance.

    Returns:
        Agent: The selected agent for the next turn.
    """
    # Retrieve messages from the group chat (currently unused).
    messages = groupchat.messages

    # Define the rotation order for speakers.
    if last_speaker == user:
        return agent_a
    elif last_speaker == agent_a:
        return agent_b
    elif last_speaker == agent_b:
        return agent_c
    # No default return if conditions not met
