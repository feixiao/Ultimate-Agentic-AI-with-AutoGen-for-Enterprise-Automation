"""
Module: Calculator Evaluation
This module defines a function to evaluate whether a calculator tool produces the expected result.
It uses Python's eval() to compute the expression and compares it with the expected result.
"""


def evaluate_calculator(expression, expected_result):
    """
    Evaluate whether the calculator produces the expected result.

    Args:
        expression (str): The arithmetic expression to evaluate.
        expected_result (numeric): The expected result of the expression.

    Returns:
        dict: A dictionary containing:
            - 'correct': True if computed result matches the expected result.
            - 'computed_result': The result computed from the expression.
            - 'expected_result': The expected result provided.
            - 'error': (optional) Error message if evaluation fails.
    """
    try:
        # Compute the result using eval()
        computed_result = eval(expression)
        # Compare computed result with expected_result and return the evaluation
        return {
            "correct": computed_result == expected_result,
            "computed_result": computed_result,
            "expected_result": expected_result,
        }
    except Exception as e:
        # Return error details in case of an exception during evaluation
        return {"error": str(e)}


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Test the evaluate_calculator function with a sample expression
    test_result = evaluate_calculator("2 + 2", 4)
    # Print the test result to the console
    print(test_result)
