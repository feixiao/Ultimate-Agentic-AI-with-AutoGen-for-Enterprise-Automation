"""
agent_rag.py
------------
Purpose: Implements a Retrieval-Augmented Generation (RAG) agent.
This module sets up the RAG tool, assistant agent, and user proxy to process queries using a knowledge base.
"""

import os
import sys
from typing import Dict, List

import autogen
from tool_rag import SimpleRAGTool

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings (e.g., llm_config) from model_config module
from model_config import *

# Initialize the RAG tool for retrieving documents and generating responses
rag = SimpleRAGTool()

# Create the assistant agent for handling queries with the RAG system
assistant = autogen.AssistantAgent(
    name="rag_assistant",
    llm_config={
        **llm_config,
        "functions": [
            {
                "name": "search_and_generate",
                "description": "Search knowledge base and generate response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's query"}
                    },
                    "required": ["query"],
                },
            }
        ],
    },
    system_message="""You are a knowledgeable assistant with access to a RAG system.
For user queries:
1. Use the search_and_generate function to find and synthesize relevant information.
2. Present information clearly and cite sources when possible.
3. If no relevant information is found, say so clearly.
Always end your response with TERMINATE.""",
)

# Create the user proxy agent to simulate admin interaction
user = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg.get("content")
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
)


# Register the RAG function for processing queries
@user.register_for_execution()
def search_and_generate(query: str) -> str:
    """Search the knowledge base and generate a response.

    Args:
        query (str): The user's query.

    Returns:
        str: The generated response from the RAG tool.
    """
    response = rag.run(query)
    return response


if __name__ == "__main__":
    # Sample documents to populate the knowledge base
    documents = [
        "The company's return policy allows returns within 30 days of purchase with original receipt.",
        "Our health insurance policy covers dental and vision after 90 days of employment.",
        "The office is open Monday through Friday, 9 AM to 5 PM Eastern Time.",
        "Customer support can be reached 24/7 via email at support@company.com.",
        "All employees are eligible for 401k matching up to 5% after 6 months of employment.",
    ]

    print("Adding documents to knowledge base...")
    rag.add_documents(documents)

    # Test queries to validate the RAG system functionality
    test_queries = [
        "What are the company's working hours?",
        "What's the vacation policy?",  # Query not covered by provided documents
    ]

    print("\nProcessing test queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        # Initiate chat between the assistant agent and the user proxy for the current query
        user.initiate_chat(assistant, message=query)
        # Reset agents for the next query
        user.reset()
        assistant.reset()
        print("-" * 50)
