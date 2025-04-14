"""
Module: Multi-turn Evaluation
This module simulates a multi-turn conversation to evaluate whether an AI retains context using a QA pipeline.
"""

from transformers import pipeline

# -----------------------------------------------------------------------------
# Initialize Question Answering Pipeline
# -----------------------------------------------------------------------------
# Create a QA pipeline using a pre-trained model (deepset/roberta-base-squad2)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


# -----------------------------------------------------------------------------
# Function: test_context_retention
# -----------------------------------------------------------------------------
def test_context_retention(conversation_history, follow_up_question):
    """
    Simulates a multi-turn conversation and evaluates whether the AI retains context.

    Args:
        conversation_history (list): List of conversation turns as strings.
        follow_up_question (str): The follow-up question to ask based on the conversation history.

    Returns:
        str: The answer provided by the QA pipeline.
    """
    # Combine the conversation history into a single context string
    context = " ".join(conversation_history)

    # Use the QA pipeline to answer the follow-up question based on the context
    result = qa_pipeline(question=follow_up_question, context=context)

    # Return the answer from the result dictionary
    return result["answer"]


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define a sample conversation history
    conversation = [
        "User: Who wrote the theory of relativity?",
        "AI: Albert Einstein developed the theory of relativity.",
    ]

    # Define a follow-up question
    follow_up_q1 = "When did he publish it?"

    # Evaluate the follow-up question using the conversation history
    response = test_context_retention(conversation, follow_up_q1)

    # Print the follow-up question and the obtained answer
    print(f"Question: {follow_up_q1}; Answer: {response}")
