"""
Module: Group Chat Conditional Sequence Pattern
This module demonstrates a conditional multi-agent group chat system.
Agents include sentiment analysis, topic classification, and summarization.
"""

import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import necessary classes from autogen library
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    register_function,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

# Append project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings from model_config
from model_config import *

# -----------------------------------------------------------------------------
# Agent Prompt Functions
# -----------------------------------------------------------------------------


def get_sentiment_agent_prompt() -> str:
    """
    Returns the prompt for the sentiment analysis agent.
    Classifies user reviews as positive, negative, or neutral.
    """
    return """
    Your job is to classify the sentiment of a user review and output a single sentiment label.
    The sentiment categories must be either "positive", "negative", or "neutral".
    Do not output anything except the sentiment. No additional commentary.
    """


def get_topic_agent_prompt() -> str:
    """
    Returns the prompt for the topic classification agent.
    Determines the main topic of a user review.
    """
    return """
    Your job is to classify a user review into its main topic and output a single topic label.
    Example negative topics: "Used product", "doesnt work", "broken item", "late delivery".
    Example positive topics: "Excellent", "brand new", "works great", "good craftsmanship".
    Do not output anything except the main topic. No additional commentary.
    """


def get_summarizer_agent_prompt() -> str:
    """
    Returns the prompt for the summarization agent.
    Provides a brief summary of the customer inquiry.
    """
    return """
    In a brief, natural language sentence, describe the customer inquiry.
    Include in quotes any category tags that have been assigned.
    """


# -----------------------------------------------------------------------------
# Initialize Agents
# -----------------------------------------------------------------------------

# User proxy agent acting as the admin initiator
user = UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg.get("content")
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Sentiment analysis agent
sentiment_agent = AssistantAgent(
    name="Sentiment",
    system_message=get_sentiment_agent_prompt(),
    llm_config=llm_config,
)

# Topic classification agent
topic_agent = AssistantAgent(
    name="Topic",
    system_message=get_topic_agent_prompt(),
    llm_config=llm_config,
)

# Summarization agent
summarize_agent = AssistantAgent(
    name="Summarize",
    system_message=get_summarizer_agent_prompt(),
    llm_config=llm_config,
)

# -----------------------------------------------------------------------------
# Custom Speaker Selection Function
# -----------------------------------------------------------------------------


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """
    Determines the next speaker based on the last speaker and message content.

    Args:
        last_speaker: The agent who spoke last.
        groupchat: The current group chat instance.

    Returns:
        The next agent to speak.
    """
    messages = groupchat.messages

    # If the user initiated, next is the sentiment agent
    if last_speaker == user:
        return sentiment_agent
    # After sentiment analysis, decide based on sentiment value
    elif last_speaker == sentiment_agent:
        last_message = messages[-1]["content"]
        if "negative" in last_message.lower():
            return topic_agent
        else:
            return summarize_agent
    # After topic classification, move to summarization
    elif last_speaker == topic_agent:
        return summarize_agent
    # Optionally, a default case can be added


# -----------------------------------------------------------------------------
# Group Chat Initialization
# -----------------------------------------------------------------------------

# Create group chat using custom speaker selection function
groupchat = GroupChat(
    agents=[sentiment_agent, topic_agent, summarize_agent],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

# -----------------------------------------------------------------------------
# Group Chat Manager Setup
# -----------------------------------------------------------------------------

# Initialize group chat manager to handle interactions
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# -----------------------------------------------------------------------------
# Execution Example
# -----------------------------------------------------------------------------

# Sample customer review for demonstration
customer_review = "I ordered a juicer and grinder and it is too bad.."

print(f"\n## Customer Query: {customer_review}")
print(
    "\n## Sequential Multi Agent Pattern, this flow will be (sentiment >> topic >> summarize) no matter what...\n"
)
print("##################\n")

# Initiate chat with the customer review
user.initiate_chat(
    manager,
    message=customer_review,
)
