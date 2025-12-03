from collections import deque


class SessionMemory:
    """
    Implements a message queue for session-based memory.

    Attributes:
        history (deque): A bounded queue storing messages as dictionaries.
    """

    def __init__(self, max_messages: int = 10):
        # Initialize a deque with a maximum length to store messages
        self.history = deque(maxlen=max_messages)

    def add_message(self, role: str, message: str) -> None:
        """
        Add a message to the session history.

        Args:
            role (str): Role of the sender (e.g., 'User', 'AI').
            message (str): The message content.
        """
        self.history.append({"role": role, "message": message})

    def get_context(self) -> str:
        """
        Retrieve the session context as a formatted string.

        Returns:
            str: A string representing the conversation history.
        """
        return "\n".join([f"{msg['role']}: {msg['message']}" for msg in self.history])


# Example usage of SessionMemory
if __name__ == "__main__":
    session_memory = SessionMemory(max_messages=5)
    session_memory.add_message("User", "Hello!")
    session_memory.add_message("AI", "Hi! How can I assist you?")

    # Print the current session history
    print("Session History:\n", session_memory.get_context())


# Implement a Summarizer Based Memory


def summarize_chat(history: str) -> str:
    """
    Generate a summary of the chat to reduce token usage while retaining key points.

    Args:
        history (str): The full conversation history as a string.

    Returns:
        str: A summarized version of the conversation.
    """
    # This is a placeholder summarization function
    return "Summary: User asked about AI memory. AI provided an overview of session-based memory."


# Replace full conversation history with a summary when token limit is reached
session_summary = summarize_chat(session_memory.get_context())
print("\nChat Summary:\n", session_summary)


# Implement a User Preference Example


class UserContextCache:
    """
    Caches user preferences for context-specific configurations.

    Attributes:
        preferences (dict): A dictionary storing user preferences.
    """

    def __init__(self):
        # Initialize an empty dictionary to store user preferences
        self.preferences = {}

    def update_preference(self, key: str, value: str) -> None:
        """
        Update a user preference.

        Args:
            key (str): The preference key.
            value (str): The preference value.
        """
        self.preferences[key] = value

    def get_preference(self, key: str) -> str:
        """
        Retrieve a user preference.

        Args:
            key (str): The preference key.

        Returns:
            str: The preference value or None if not set.
        """
        return self.preferences.get(key, None)


# Example usage of UserContextCache
user_cache = UserContextCache()
user_cache.update_preference("response_style", "detailed")
print(
    "\nUser Preference for response_style:", user_cache.get_preference("response_style")
)
