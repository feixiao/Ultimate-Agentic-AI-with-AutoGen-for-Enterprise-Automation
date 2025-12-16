from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json


class ContextWindowManager:
    """
    Manages a sliding context window for LLM interactions with priority-based eviction.

    Attributes:
        max_tokens (int): Maximum tokens allowed in the context window.
        priority_threshold (float): Minimum priority score to accept new context.
        context_buffer (list): Buffer storing context items with metadata.

    Example:
        manager = ContextWindowManager(max_tokens=4096)
        manager.add_context({"role": "user", "content": "Important information"}, token_count=10)
    """

    def __init__(self, max_tokens: int = 4096, priority_threshold: float = 0.7):
        # Initialize maximum token limit and priority threshold
        self.max_tokens = max_tokens
        self.priority_threshold = priority_threshold

        # Initialize context buffer and last access time tracking
        self.context_buffer = []
        self._last_access_times = {}  # Format: {context_id: datetime object}

    def add_context(self, new_context: dict, token_count: int) -> bool:
        """
        Adds new context to the buffer if it meets the priority threshold.

        Args:
            new_context (dict): Context data to add.
            token_count (int): Number of tokens in the new context.

        Returns:
            bool: True if context was added; False if rejected.
        """
        # Calculate the priority score for the new context
        priority_score = self._calculate_priority(new_context)

        # Check if the calculated priority meets the required threshold
        if priority_score >= self.priority_threshold:
            # Ensure there is enough space by managing the buffer
            self._manage_buffer(token_count)

            # Generate a unique identifier for the new context
            context_id = self._generate_context_id(new_context)

            # Create a new context item with metadata
            context_item = {
                "id": context_id,
                "content": new_context,
                "priority": priority_score,
                "tokens": token_count,
                "timestamp": datetime.now().isoformat(),
            }

            # Add the new context item to the buffer
            self.context_buffer.append(context_item)
            # Record the current access time for this context
            self._last_access_times[context_id] = datetime.now()
            return True

        # Return False if the context did not meet the priority threshold
        return False

    def _manage_buffer(self, required_tokens: int) -> None:
        """
        Ensures there is enough space in the buffer by evicting low-priority items.

        Args:
            required_tokens (int): Tokens required for new content.
        """
        # Evict items until total tokens plus required tokens fit within max_tokens
        while self._get_total_tokens() + required_tokens > self.max_tokens:
            if not self._evict_lowest_priority():
                # Raise error if unable to free sufficient space
                raise ValueError("Cannot free enough space in context buffer")

    def _calculate_priority(self, context: dict) -> float:
        """
        Calculates a priority score for the given context.
        核心目的: 为每条上下文计算一个介于 0 到 1 的“优先级分数”，用于衡量其重要性。
        应用场景: 决定是否接纳新上下文、按优先级排序返回上下文、在空间不足时优先淘汰低优先级项。

        Args:
            context (dict): Context data to evaluate.

        Returns:
            float: Priority score between 0 and 1.
        """
        # Factor 1: Content length weight (longer content might indicate higher importance)
        content_length = len(json.dumps(context))
        length_score = min(content_length / 1000, 1.0)  # Normalized to [0, 1]

        # Factor 2: Recency weight (more recent content gets higher score)
        if "timestamp" in context:
            age_seconds = (
                datetime.now() - datetime.fromisoformat(context["timestamp"])
            ).total_seconds()
            recency_score = np.exp(-age_seconds / 3600)  # Exponential decay per hour
        else:
            recency_score = 1.0

        # Factor 3: Content type weight based on role
        type_weights = {"system": 1.0, "user": 0.9, "assistant": 0.7, "metadata": 0.5}
        type_score = type_weights.get(context.get("role", "metadata"), 0.5)

        # Factor 4: Custom importance flag from context
        importance_score = float(context.get("important", 0.5))

        # Combine factors with predefined weights
        priority_score = (
            length_score * 0.2
            + recency_score * 0.3
            + type_score * 0.3
            + importance_score * 0.2
        )

        # Ensure the priority score is between 0 and 1
        return min(max(priority_score, 0.0), 1.0)

    def _evict_lowest_priority(self) -> bool:
        """
        Evicts the lowest priority context item from the buffer.

        Returns:
            bool: True if an item was evicted; False if buffer is empty.
        """
        if not self.context_buffer:
            return False

        # Identify the index of the item with the lowest priority and earliest access time
        lowest_idx = min(
            range(len(self.context_buffer)),
            key=lambda i: (
                self.context_buffer[i]["priority"],
                self._last_access_times[self.context_buffer[i]["id"]],
            ),
        )

        # Remove the identified item from the buffer
        evicted_item = self.context_buffer.pop(lowest_idx)
        # Remove its access time record
        del self._last_access_times[evicted_item["id"]]
        return True

    def _get_total_tokens(self) -> int:
        """
        Calculates the total number of tokens currently stored in the buffer.

        Returns:
            int: Total token count.
        """
        return sum(item["tokens"] for item in self.context_buffer)

    def _generate_context_id(self, context: dict) -> str:
        """
        Generates a unique identifier for a context item.

        Args:
            context (dict): Context data.

        Returns:
            str: Unique identifier string.
        """
        # Create a hash from the sorted JSON representation of the context
        content_hash = hash(json.dumps(context, sort_keys=True))
        # Append the current timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"ctx_{content_hash}_{timestamp}"

    def get_context(self, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves the current context buffer, optionally limited to the top k items by priority.

        Args:
            k (Optional[int]): Number of highest priority items to return.

        Returns:
            List[Dict]: Context items sorted by descending priority.
        """
        # Sort the context items by priority (highest first)
        sorted_context = sorted(
            self.context_buffer, key=lambda x: x["priority"], reverse=True
        )

        # Update last access times for the returned items
        for item in sorted_context[:k]:
            self._last_access_times[item["id"]] = datetime.now()

        # Return the full sorted list or the top k items
        return sorted_context if k is None else sorted_context[:k]

    def clear_buffer(self) -> None:
        """
        Clears the context buffer and resets access time tracking.
        """
        self.context_buffer = []
        self._last_access_times = {}


def main():
    """
    Main function demonstrating the usage of ContextWindowManager.
    """
    # Initialize the context manager with token limit and priority threshold
    manager = ContextWindowManager(max_tokens=4096, priority_threshold=0.7)

    # Example: Add system context with high importance
    context1 = {
        "role": "system",
        "content": "Important system configuration",
        "important": 1.0,
    }
    manager.add_context(context1, token_count=20)

    # Example: Add user context
    context2 = {"role": "user", "content": "User query about system status"}
    manager.add_context(context2, token_count=15)

    # Retrieve and print the top 5 context items by priority
    top_context = manager.get_context(k=5)
    print("Top context items:")
    for item in top_context:
        print(item)


if __name__ == "__main__":
    main()
