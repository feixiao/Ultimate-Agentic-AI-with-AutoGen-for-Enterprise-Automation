import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
from pathlib import Path

import os, sys

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings from model_config module
from model_config import *


class AutoGenMemory:
    """
    Manages short-term memory for AutoGen agents by storing conversation history
    on the local file system. Provides methods to retrieve, update, and manage
    conversation history with features such as:
      - Persistent storage of conversation history
      - Conversation threading and grouping
      - Memory pruning and management
      - Context injection for new conversations
    """

    def __init__(
        self,
        storage_path: str = "autogen_memory",
        max_conversations: int = 10,
        max_messages_per_conversation: int = 50,
    ):
        """
        Initialize the memory manager.

        Args:
            storage_path: Directory path for storing conversation history.
            max_conversations: Maximum number of conversations to store.
            max_messages_per_conversation: Maximum messages per conversation.
        """
        self.storage_path = Path(storage_path)
        self.max_conversations = max_conversations
        self.max_messages = max_messages_per_conversation

        # Create the storage directory if it doesn't exist.
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize conversation index from disk or as an empty dictionary.
        self.index_path = self.storage_path / "conversation_index.json"
        self.conversation_index = self._load_conversation_index()

    def store_conversation(
        self,
        messages: List[Dict],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store a conversation or update an existing one.

        Args:
            messages: List of message dictionaries representing the conversation.
            conversation_id: Optional ID for an existing conversation; if not provided, generated automatically.
            metadata: Optional metadata for the conversation.

        Returns:
            str: The conversation ID.
        """
        # Generate conversation ID if not provided.
        if not conversation_id:
            conversation_id = self._generate_conversation_id(messages)

        # Prepare conversation data by keeping only the most recent messages.
        conversation_data = {
            "conversation_id": conversation_id,
            "messages": messages[-self.max_messages :],  # Retain only recent messages.
            "metadata": metadata or {},
            "last_updated": datetime.now().isoformat(),
            "message_count": len(messages),
        }

        # Save the conversation data to a JSON file.
        conversation_path = self.storage_path / f"{conversation_id}.json"
        with open(conversation_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2)

        # Update the conversation index with the new conversation data.
        self._update_conversation_index(conversation_id, conversation_data)

        # Prune old conversations if the maximum limit is exceeded.
        self._prune_conversations()

        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Retrieve a stored conversation.

        Args:
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            Optional[Dict]: The conversation data if found; otherwise, None.
        """
        conversation_path = self.storage_path / f"{conversation_id}.json"
        if not conversation_path.exists():
            return None

        with open(conversation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_recent_messages(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent messages from a conversation.

        Args:
            conversation_id: The conversation ID.
            limit: Maximum number of messages to retrieve.

        Returns:
            List[Dict]: A list of the most recent messages.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        # Return only the last 'limit' messages.
        return conversation["messages"][-limit:]

    def build_context_for_agent(
        self, conversation_id: str, max_tokens: Optional[int] = None
    ) -> str:
        """
        Build a context string for initializing an AutoGen agent.

        Args:
            conversation_id: The conversation ID.
            max_tokens: Optional maximum token limit (currently not utilized).

        Returns:
            str: A formatted context string derived from the conversation.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return ""

        # Build a context string from conversation messages.
        context_parts = []
        for msg in conversation["messages"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search conversations using simple keyword matching.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List[Dict]: A list of matching conversations with basic details.
        """
        results = []
        query = query.lower()

        # Iterate over the conversation index and perform keyword matching.
        for conv_id in self.conversation_index:
            conversation = self.get_conversation(conv_id)
            if not conversation:
                continue

            # Convert messages to lower-case JSON string for matching.
            content = json.dumps(conversation["messages"]).lower()
            if query in content:
                results.append(
                    {
                        "conversation_id": conv_id,
                        "last_updated": conversation["last_updated"],
                        "message_count": conversation["message_count"],
                        "metadata": conversation["metadata"],
                    }
                )

            # Stop if the number of results meets the specified limit.
            if len(results) >= limit:
                break

        return results

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: The ID of the conversation to delete.

        Returns:
            bool: True if deletion was successful; otherwise, False.
        """
        conversation_path = self.storage_path / f"{conversation_id}.json"
        if not conversation_path.exists():
            return False

        # Remove the conversation file.
        conversation_path.unlink()

        # Update the conversation index after deletion.
        if conversation_id in self.conversation_index:
            del self.conversation_index[conversation_id]
            self._save_conversation_index()

        return True

    def _generate_conversation_id(self, messages: List[Dict]) -> str:
        """
        Generate a unique conversation ID based on the messages' content.

        Args:
            messages: A list of message dictionaries.

        Returns:
            str: The generated conversation ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        content_hash = hashlib.md5(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"conv_{timestamp}_{content_hash}"

    def _load_conversation_index(self) -> Dict:
        """
        Load the conversation index from disk.

        Returns:
            Dict: The conversation index, or an empty dictionary if not found.
        """
        if not self.index_path.exists():
            return {}

        with open(self.index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_conversation_index(self) -> None:
        """
        Save the conversation index to disk.
        """
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_index, f, indent=2)

    def _update_conversation_index(
        self, conversation_id: str, conversation_data: Dict
    ) -> None:
        """
        Update the conversation index with new or updated conversation data.

        Args:
            conversation_id: The conversation ID.
            conversation_data: The conversation data.
        """
        self.conversation_index[conversation_id] = {
            "last_updated": conversation_data["last_updated"],
            "message_count": conversation_data["message_count"],
        }
        self._save_conversation_index()

    def _prune_conversations(self) -> None:
        """
        Remove the oldest conversations if the number of conversations exceeds the maximum limit.
        """
        if len(self.conversation_index) <= self.max_conversations:
            return

        # Sort conversations by the 'last_updated' timestamp.
        sorted_conversations = sorted(
            self.conversation_index.items(), key=lambda x: x[1]["last_updated"]
        )

        # Identify conversations to remove (oldest ones).
        conversations_to_remove = sorted_conversations[
            : len(self.conversation_index) - self.max_conversations
        ]

        # Delete the identified conversations.
        for conv_id, _ in conversations_to_remove:
            self.delete_conversation(conv_id)


### Usage Example ###
from autogen import AssistantAgent, UserProxyAgent

# Initialize the memory manager with specified storage path and limits.
memory = AutoGenMemory(
    storage_path="./conversation_history",
    max_conversations=100,
    max_messages_per_conversation=50,
)

# Create a new conversation or load an existing one.
conversation_id = (
    "existing_conversation_id"  # Optional: Replace with a valid ID if available.
)
conversation_data = memory.get_conversation(conversation_id)

# Initialize the user proxy with context from memory.
initial_context = (
    memory.build_context_for_agent(conversation_id) if conversation_id else ""
)

# Create agents with memory-aware configurations.
user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message=f"Previous context:\n{initial_context}\n\nContinue the conversation considering the above context.",
    code_execution_config={"work_dir": "coding"},
)

assistant = AssistantAgent(name="assistant", llm_config=llm_config)

# Start the conversation.
user_proxy.initiate_chat(assistant, message="Continue our previous discussion")

# After the conversation, store the messages.
messages = user_proxy.chat_messages[assistant.name]
memory.store_conversation(
    messages=messages,
    conversation_id=conversation_id,
    metadata={"topic": "code discussion", "participants": ["user_proxy", "assistant"]},
)
