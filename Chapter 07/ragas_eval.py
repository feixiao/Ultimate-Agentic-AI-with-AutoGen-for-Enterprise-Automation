"""
Module: RAGAs Evaluation
This module evaluates generated responses against ground truth using various metrics,
such as faithfulness, answer relevance, and contextual precision.
"""

# -----------------------------------------------------------------------------
# Import Dependencies
# -----------------------------------------------------------------------------
import ragas
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    contextual_precision,
)
from ragas.evaluator import evaluate

# -----------------------------------------------------------------------------
# Define Test Data
# -----------------------------------------------------------------------------
# A list of dictionaries containing test cases for evaluation.
data = [
    {
        "question": "Who discovered penicillin?",
        "ground_truth": "Alexander Fleming discovered penicillin in 1928.",
        "generated_response": (
            "Penicillin was first discovered by Alexander Fleming in the early 20th century."
        ),
        "retrieved_context": "Alexander Fleming discovered penicillin in 1928.",
    }
]

# -----------------------------------------------------------------------------
# Run Evaluation
# -----------------------------------------------------------------------------
# Evaluate the generated response using the specified metrics:
#   - faithfulness: Checks if the generated response adheres to the factual context.
#   - answer_relevance: Measures how relevant the answer is to the question.
#   - contextual_precision: Evaluates the precision of contextual details.
results = evaluate(
    dataset=data, metrics=[faithfulness, answer_relevance, contextual_precision]
)

# -----------------------------------------------------------------------------
# Output Results
# -----------------------------------------------------------------------------
# Print the evaluation results to the console.
print(results)
