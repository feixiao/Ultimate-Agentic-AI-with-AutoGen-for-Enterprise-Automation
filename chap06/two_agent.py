"""
Module: Two-Agent Conversation with Critique
This module demonstrates a two-agent system:
- Answerer: Provides detailed answers to questions.
- Critic: Evaluates the provided answers for accuracy and clarity.
A user proxy (Admin) coordinates the conversation.
"""

from autogen import AssistantAgent, UserProxyAgent
import os
import sys

# -----------------------------------------------------------------------------
# Module Setup
# -----------------------------------------------------------------------------
# Append the project root to sys.path for module discovery.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import configuration settings from the chap06 package for direct script execution.
from chap06 import llm_config

# -----------------------------------------------------------------------------
# Agent Initialization
# -----------------------------------------------------------------------------
# Create the answering agent with a detailed answer prompt.
answerer = AssistantAgent(
    name="Answerer",
    llm_config=llm_config,
    system_message=(
        "You are a knowledgeable assistant who provides detailed answers to questions. "
        "After giving the answer, end with TERMINATE in a new line."
    ),
)

# Create the critic agent with instructions for evaluating the answer.
critic = AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message=(
        "You are a critical thinker who evaluates answers for accuracy, completeness, and clarity. "
        "Provide constructive feedback and suggestions for improvement."
    ),
)

# Create a user proxy agent acting as an admin initiator.
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg.get("content")
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
)


# -----------------------------------------------------------------------------
# Conversation Function
# -----------------------------------------------------------------------------
def get_answer_with_critique(question):
    """
    Initiates a conversation to obtain an answer and then a critique for the given question.

    Args:
        question (str): The question to be answered and evaluated.
    """
    # Initiate chat with the answerer to get the answer.
    user_proxy.initiate_chat(
        answerer, message=f"Please answer this question: {question}"
    )

    # Retrieve the last message, which contains the answer.
    answer = user_proxy.last_message()

    # Initiate chat with the critic to evaluate the answer.
    user_proxy.initiate_chat(
        critic,
        message=f"Please evaluate this answer to the question '{question}':\n\n{answer}",
    )


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    question = "What are the main causes of climate change?"
    get_answer_with_critique(question)
