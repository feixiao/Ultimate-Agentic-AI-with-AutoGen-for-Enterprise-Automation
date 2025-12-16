"""
agent_math.py
-------------
Purpose: Implements a math assistant agent that uses CalculatorTool to solve mathematical problems.
Sets up the agent, registers the solve_math function, and runs test queries.
"""

import os
import sys
from typing import Dict, Any

import autogen
from tool_calculator import CalculatorTool

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import configuration settings from model_config module
from chap04 import llm_config

# Initialize CalculatorTool with the provided LLM configuration
calculator = CalculatorTool(llm_config)

# Create the assistant agent for handling math queries
assistant = autogen.AssistantAgent(
    name="math_assistant",
    llm_config={
        **llm_config,
        "functions": [
            {
                "name": "solve_math",
                "description": "Solve mathematical problems given in natural language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The mathematical query in natural language",
                        }
                    },
                    "required": ["query"],
                },
            }
        ],
    },
    system_message="""You are a helpful assistant that specializes in solving mathematical problems.
For mathematical queries:
    - Use the solve_math function to calculate the result
    - Explain your approach clearly
    - Present the result in a clear format
    - End with TERMINATE

For non-mathematical queries:
    - Provide a direct, clear answer
    - Do not mention calculation capabilities
    - End with TERMINATE

Remember: Always verify if a query needs calculation before using the solve_math function.""",
)

# Create the user proxy agent for interacting with the assistant
user = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg.get("content")
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
)


# Register the solve_math function for executing math queries
@user.register_for_execution()
def solve_math(query: str) -> str:
    """Solve a mathematical problem given in natural language.

    Args:
        query (str): The math problem expressed in natural language.

    Returns:
        str: The result and the generated arithmetic expression, or an error message.
    """
    result = calculator.run(query)

    if "error" in result:
        return f"Error: {result['error']}"

    return (
        f"Result: {result['result']}\nGenerated expression: {result['code_generated']}"
    )


if __name__ == "__main__":
    # Define test queries for the math assistant
    queries = [
        "What is the capital of France?",
        "Calculate the sum of 23.5 and 17.8",
        "What is the product of 15 and 3?",
        "Who invented the telephone?",
        "What is 25 percent of 80?",
    ]

    # Process each test query
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        # Initiate chat between the user proxy and the assistant with the current query
        user.initiate_chat(assistant, message=query)

        # Reset agents for the next query
        user.reset()
        assistant.reset()
        print("-" * 50)
