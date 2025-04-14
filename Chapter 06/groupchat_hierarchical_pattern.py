"""
Module: Group Chat Hierarchical Pattern
This module demonstrates a hierarchical multi-agent group chat system.
Agents include sentiment analysis, topic classification, technical and car product classification, and summarization.
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
    This prompt instructs the agent to classify user reviews as positive, negative, or neutral.
    """
    return """
    Your job is to classify the sentiment of a user review and output a single sentiment label.
    The sentiment categories have to be either "positive", "negative", or "neutral".
    Do not output anything except the sentiment. Do not include any additional commentary.
    """


def get_topic_agent_prompt() -> str:
    """
    Returns the prompt for the topic classification agent.
    This prompt instructs the agent to determine the main topic of a user review.
    """
    return """
    Your job is to classify a user review into its main topic and output a single topic label.
    Some example topics for negative reviews could be "Used product", "doesnt work", "broken item", "late delivery", etc.
    Some example topics for positive reviews could be "Excellent", "brand new", "works great", "good craftsmanship", etc.
    Do not output anything except the main topic. Do not include any additional commentary.
    """


def get_tech_product_agent_prompt() -> str:
    """
    Returns the prompt for the technical product classification agent.
    This prompt instructs the agent to classify a review regarding technical products like phones, tablets, laptops, etc.
    """
    return """
    Your job is to classify a user review into technical products like phone, tablets, laptops, etc.
    If the review is not about a technology product, do not do anything.
    Do not output anything except the sentiment. Do not include any additional commentary.
    """


def get_car_product_agent_prompt() -> str:
    """
    Returns the prompt for the car product classification agent.
    This prompt instructs the agent to classify a review regarding car products.
    """
    return """
    Your job is to classify a user review into car products.
    If the review is not about a car product, do not do anything.
    Do not output anything except the sentiment. Do not include any additional commentary.
    """


def get_summarizer_agent_prompt() -> str:
    """
    Returns the prompt for the summarization agent.
    This prompt instructs the agent to provide a brief summary of the customer inquiry.
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

# Initialize the technical product classification agent
tech_agent = AssistantAgent(
    name="TechnologyProducts",
    system_message=get_tech_product_agent_prompt(),
    llm_config=llm_config,
)

# Initialize the car product classification agent
car_agent = AssistantAgent(
    name="CarProducts",
    system_message=get_car_product_agent_prompt(),
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
    Determines the next speaker based on the last speaker in the conversation.

    Args:
        last_speaker: The agent who spoke last.
        groupchat: The current group chat instance containing all messages.

    Returns:
        The next agent to speak or a special command for automatic selection.
    """
    messages = groupchat.messages

    if last_speaker == user:
        # When initiated by the user, let the manager decide the domain.
        return "auto"
    if last_speaker in (tech_agent, car_agent):
        # After domain-specific agents, route to the topic agent.
        return topic_agent
    elif last_speaker == topic_agent:
        # After topic classification, route to the summarization agent.
        return summarize_agent
    # Default case: if none of the conditions match, return None.
    return None


# ---------------------------------------------------------------------------
# Group Chat Initialization
# ---------------------------------------------------------------------------

groupchat = GroupChat(
    agents=[sentiment_agent, topic_agent, summarize_agent, tech_agent, car_agent],
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
customer_review = "I ordered a tech product, the screen was broken."

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
