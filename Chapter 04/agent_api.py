import os
import sys
import json
from typing import Dict, Any

import autogen
from tool_api import APITool

# Append the project root directory to sys.path for module discovery
# This enables absolute imports (e.g., model_config) from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings from model_config module
from model_config import *


class APIAgent:
    """
    APIAgent interfaces with an external API using AutoGen agents.
    It sets up both a user proxy agent and an assistant agent to process API-related queries.
    """

    def __init__(self):
        """
        Initialize the APIAgent with LLM configuration, API tool, and agents.
        """
        # Setup LLM configuration from global settings
        self.llm_config = llm_config

        # Initialize API tool with a mock API endpoint and authentication token
        self.api_tool = APITool(
            base_url="https://api.mockcompany.com", auth_token="mock_token"
        )

        # Create the assistant agent with an embedded API function for query handling
        self.assistant = autogen.AssistantAgent(
            name="api_assistant",
            llm_config={
                **self.llm_config,
                "functions": [
                    {
                        "name": "query_api",
                        "description": "Make API requests to check order or billing status",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "endpoint": {
                                    "type": "string",
                                    "description": "API endpoint to call",
                                },
                                "method": {
                                    "type": "string",
                                    "description": "HTTP method (GET, POST, PUT, DELETE)",
                                },
                                "params": {
                                    "type": "object",
                                    "description": "Query parameters for the request",
                                },
                                "data": {
                                    "type": "object",
                                    "description": "Data payload for POST/PUT requests",
                                },
                            },
                            "required": ["endpoint", "method"],
                        },
                    }
                ],
            },
            system_message="""You are an assistant that helps users check order and billing status through API calls.

You can:
1. Check order status using GET /orders/{order_id}
2. Check billing status using GET /billing/{invoice_id}
3. Update order status using PUT /orders/{order_id}

Always validate user input before making API calls.
Format responses clearly and handle errors gracefully.
End your response with TERMINATE""",
        )

        # Create the user proxy agent, acting as an interface for administrative input
        self.user = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin.",
            is_termination_msg=lambda msg: (
                msg.get("content") is not None and "TERMINATE" in msg.get("content")
            ),
            human_input_mode="NEVER",  # Disable interactive human input
            code_execution_config=False,  # Code execution is disabled
        )

        # Register the API function so the user proxy can execute API calls
        @self.user.register_for_execution()
        def query_api(
            endpoint: str, method: str, params: Dict = None, data: Dict = None
        ) -> str:
            """
            Execute an API request using the provided parameters.

            Parameters:
                endpoint (str): The API endpoint to call.
                method (str): The HTTP method (GET, POST, PUT, DELETE).
                params (Dict, optional): Query parameters for the API request.
                data (Dict, optional): Data payload for POST/PUT requests.

            Returns:
                str: Formatted API response or an error message.
            """
            result = self.api_tool.run(
                {"endpoint": endpoint, "method": method, "params": params, "data": data}
            )

            # Check for errors in the API response and return an error message if found
            if "error" in result:
                return f"Error making API request: {result['error']}"

            # Return a formatted API response including status code and response data
            return f"API Response (Status {result['status_code']}):\n{json.dumps(result['data'], indent=2)}"

    def process_query(self, user_query: str) -> None:
        """
        Process a user query by initiating a chat between the user proxy and the assistant agent.

        Parameters:
            user_query (str): The user query to be processed.
        """
        # Initiate a chat with the assistant agent using the provided user query
        self.user.initiate_chat(self.assistant, message=user_query)

        # Reset the agents to clear the conversation state for subsequent queries
        self.user.reset()
        self.assistant.reset()


def main():
    """
    Main function to test the APIAgent by processing a series of test queries.
    """
    # Initialize the APIAgent instance
    agent = APIAgent()

    # Define a list of test queries to simulate API interactions
    test_queries = [
        "Check the status of order #12345",
        "View billing information for invoice INV-789",
        "Update order #12345 status to shipped",
        "Get all orders from last week",  # Complex query
    ]

    # Process each query and display the output
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        agent.process_query(query)
        print("-" * 50)


if __name__ == "__main__":
    main()
