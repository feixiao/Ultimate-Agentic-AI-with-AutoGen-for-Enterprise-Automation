"""
Module: Llama Index Response Evaluation
This module uses LlamaIndex to evaluate the quality of a retrieved document
and an AI-generated response based on a given query.
"""

# -----------------------------------------------------------------------------
# Import Dependencies
# -----------------------------------------------------------------------------
from llama_index.evaluation import ResponseEvaluator
from llama_index import ServiceContext, LLMPredictor
from langchain.chat_models import ChatOpenAI

# -----------------------------------------------------------------------------
# Initialize Language Model and Service Context
# -----------------------------------------------------------------------------
# Initialize a ChatOpenAI model (using GPT-4) for response evaluation.
llm_predictor = LLMPredictor(llm=ChatOpenAI(model="gpt-4"))

# Create a service context with default settings, utilizing the initialized LLM predictor.
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Instantiate the evaluator with the provided service context.
evaluator = ResponseEvaluator(service_context)


# -----------------------------------------------------------------------------
# Function: evaluate_rag_response
# -----------------------------------------------------------------------------
def evaluate_rag_response(query, retrieved_text, generated_response):
    """
    Evaluates the quality of an AI-generated response given a query and its context.

    Args:
        query (str): The user's question or prompt.
        retrieved_text (str): The context or document retrieved for the query.
        generated_response (str): The AI-generated response to be evaluated.

    Returns:
        dict: A dictionary containing evaluation scores and feedback.
    """
    # Use the evaluator to assess the response quality.
    scores = evaluator.evaluate(
        query=query, context=retrieved_text, response=generated_response
    )
    return scores


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define sample inputs for demonstration.
    query = "Who discovered gravity?"
    retrieved_text = (
        "Sir Isaac Newton formulated the law of universal gravitation in 1687."
    )
    generated_response = "Gravity was discovered by Isaac Newton in the 17th century."

    # Evaluate the response quality using the provided inputs.
    evaluation_scores = evaluate_rag_response(query, retrieved_text, generated_response)

    # Output the evaluation results.
    print(evaluation_scores)
