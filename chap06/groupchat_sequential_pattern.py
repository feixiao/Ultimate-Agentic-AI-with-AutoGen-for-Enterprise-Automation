"""
Module: Group Chat Sequential Pattern
This module demonstrates a sequential multi-agent group chat system where each agent performs a specific task:
- Sentiment analysis
- Topic classification
- Summarization
"""

import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import necessary classes and functions from the autogen library
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    register_function,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings from model_config
from model_config import *

# ---------------------------------------------------------------------------
# Agent Prompt Functions
# ---------------------------------------------------------------------------


def get_sentiment_agent_prompt() -> str:
    """
    Returns the prompt for the sentiment analysis agent.
    The agent classifies a user review as positive, negative, or neutral.
    """
    return """
    Your job is to classify the sentiment of a user review and output a single sentiment label.
    The sentiment categories must be either "positive", "negative", or "neutral".
    Do not output anything except the sentiment. No additional commentary.
    """


def get_topic_agent_prompt() -> str:
    """
    Returns the prompt for the topic classification agent.
    The agent identifies the main topic of a user review.
    """
    return """
    Your job is to classify a user review into its main topic and output a single topic label.
    Example negative topics: "Used product", "doesnt work", "broken item", "late delivery", etc.
    Example positive topics: "Excellent", "brand new", "works great", "good craftsmanship", etc.
    Do not output anything except the main topic. No additional commentary.
    """


def get_summarizer_agent_prompt() -> str:
    """
    Returns the prompt for the summarization agent.
    The agent provides a brief summary of the customer inquiry.
    """
    return """
    In a brief, natural language sentence, describe the customer inquiry.
    Include in quotes any category tags that have been assigned.
    """


# ---------------------------------------------------------------------------
# Initialize Agents
# ---------------------------------------------------------------------------

# Create a user proxy agent acting as an admin initiator
user = UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg.get("content")
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Initialize the sentiment analysis agent
sentiment_agent = AssistantAgent(
    name="Sentiment",
    system_message=get_sentiment_agent_prompt(),
    llm_config=llm_config,
)

# Initialize the topic classification agent
topic_agent = AssistantAgent(
    name="Topic",
    system_message=get_topic_agent_prompt(),
    llm_config=llm_config,
)

# Initialize the summarization agent
summarize_agent = AssistantAgent(
    name="Summarize",
    system_message=get_summarizer_agent_prompt(),
    llm_config=llm_config,
)

# ---------------------------------------------------------------------------
# Custom Speaker Selection Function
# ---------------------------------------------------------------------------


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """
    Determines the next agent to speak based on the last speaker.

    Args:
        last_speaker: The agent who spoke last.
        groupchat: The current group chat instance containing all messages.

    Returns:
        The next agent to speak.
    """
    messages = groupchat.messages

    # If the user initiated the conversation, start with the sentiment agent
    if last_speaker == user:
        return sentiment_agent
    # After sentiment analysis, move to topic classification
    elif last_speaker == sentiment_agent:
        return topic_agent
    # After topic classification, proceed to summarization
    elif last_speaker == topic_agent:
        return summarize_agent

    # If no condition matches, return None
    return None


# ---------------------------------------------------------------------------
# Group Chat Initialization
# ---------------------------------------------------------------------------

groupchat = GroupChat(
    agents=[sentiment_agent, topic_agent, summarize_agent],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

# ---------------------------------------------------------------------------
# Group Chat Manager Setup
# ---------------------------------------------------------------------------

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# ---------------------------------------------------------------------------
# Execution Example
# ---------------------------------------------------------------------------

# Sample customer review for demonstration
customer_review = "I ordered a juicer and grinder and it is awesome.."

print(f"\n## Customer Query: {customer_review}")
print("\n## Sequential Multi Agent Pattern: sentiment >> topic >> summarize\n")
print("##################\n")

# Initiate chat with the customer review
user.initiate_chat(
    manager,
    message=customer_review,
)
