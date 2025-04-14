from typing import Dict, List, Any
import tiktoken


def token_count(text: str, encoder: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in a given text using the specified encoder.

    Args:
        text (str): The input text to tokenize.
        encoder (str): The tokenizer model to use (default is 'cl100k_base').

    Returns:
        int: The total token count.
    """
    # Get the encoding object based on the provided encoder
    encoding = tiktoken.get_encoding(encoder)
    # Encode the text and return the number of tokens
    return len(encoding.encode(text))


def truncate_context(text: str, max_tokens: int, encoder: str = "cl100k_base") -> str:
    """
    Truncate the input text to ensure it fits within a specified token limit.

    Args:
        text (str): The input text to truncate.
        max_tokens (int): Maximum allowed number of tokens.
        encoder (str): The tokenizer model to use (default is 'cl100k_base').

    Returns:
        str: The truncated text that fits within the token limit.
    """
    # Retrieve the encoding for the specified model
    encoding = tiktoken.encoding_for_model(encoder)
    # Encode the text into tokens
    tokens = encoding.encode(text)

    # Truncate tokens list to the maximum allowed tokens
    truncated_tokens = tokens[:max_tokens]

    # Decode the truncated tokens back into a string
    truncated_text = encoding.decode(truncated_tokens)

    return truncated_text


def completion_with_llm(query: str):
    """
    Run text generation using an LLM (e.g., OpenAI or Llama models).

    This is a placeholder function to integrate with an LLM.

    Args:
        query (str): The input query string for text generation.

    Returns:
        Any: Generated output from the LLM. Currently returns None.
    """
    # TODO: Implement LLM integration for text generation
    return None


# Default context management


def agent_context_management(prompt: str, context: str, max_tokens: int = 4096) -> str:
    """
    Manage agent context by combining context and prompt while respecting token limits.

    This function:
    - Combines a context string and a prompt.
    - Checks if the combined input exceeds a maximum token limit.
    - Truncates the input if necessary.
    - Generates a completion using an LLM.

    Args:
        prompt (str): The prompt for the agent.
        context (str): The context information to include.
        max_tokens (int): Maximum allowed token count (default: 4096).

    Returns:
        str: The output from the LLM after processing the combined input.
    """
    # Combine context and prompt with a clear separator
    formatted_input = f"Context:\n{context}\n\nPrompt:\n{prompt}"

    # Calculate estimated token count of the combined input
    estimated_tokens = token_count(formatted_input)

    # If token count exceeds maximum allowed, truncate the combined input
    if estimated_tokens > max_tokens:
        formatted_input = truncate_context(formatted_input, max_tokens)

    # Pass the processed input to the LLM for completion generation
    return completion_with_llm(formatted_input)
