from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import torch
from torch.nn.functional import cosine_similarity
import logging


# Define an enumeration for priority levels
class PriorityLevel(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    ARCHIVAL = 0


# Dataclass to represent a context item with its content, metadata, and priority details
@dataclass
class ContextItem:
    content: Union[str, dict]  # The context content (text or structured data)
    metadata: dict  # Additional metadata
    embedding: Optional[np.ndarray] = None  # Pre-computed embedding vector
    priority_score: float = 0.0  # Numeric score representing priority
    last_accessed: datetime = None  # Timestamp of the last access
    creation_time: datetime = None  # Timestamp when the context was created
    access_count: int = 0  # Number of times accessed


class ContextPrioritizer:
    """
    Advanced context prioritization system using embeddings and multi-factor scoring.

    Features:
    - Semantic similarity scoring using embeddings
    - Time-based decay of priority scores
    - Usage pattern analysis
    - Adaptive thresholding
    - Batch processing capabilities
    """

    def __init__(
        self,
        embedding_model,
        embedding_dimension: int = 768,
        max_context_items: int = 1000,
        base_decay_rate: float = 0.1,
        similarity_threshold: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Initialize model parameters and storage structures
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.max_context_items = max_context_items
        self.base_decay_rate = base_decay_rate
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Dictionary to store context items with unique IDs
        self.context_items: Dict[str, ContextItem] = {}
        # Priority buckets to group context IDs by PriorityLevel
        self.priority_buckets: Dict[PriorityLevel, List[str]] = defaultdict(list)

        # Cache for storing computed embeddings
        self.embedding_cache = {}

        # Setup logging for debugging and traceability
        self.logger = logging.getLogger(__name__)

    def add_context(
        self,
        content: Union[str, dict],
        metadata: Optional[dict] = None,
        initial_priority: PriorityLevel = PriorityLevel.MEDIUM,
    ) -> str:
        """
        Add a new context item with an initial priority.

        Args:
            content: The context content (text or structured data)
            metadata: Optional metadata for the context
            initial_priority: Starting priority level

        Returns:
            str: Unique identifier for the added context item
        """
        # Evict lowest priority item if maximum capacity is reached
        if len(self.context_items) >= self.max_context_items:
            self._evict_lowest_priority()

        # Generate a unique context ID based on content and current timestamp
        context_id = self._generate_context_id(content)

        # Generate embedding for the provided content
        embedding = self._generate_embedding(content)

        # Create a new ContextItem with initial values
        context_item = ContextItem(
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            priority_score=initial_priority.value,
            last_accessed=datetime.now(),
            creation_time=datetime.now(),
            access_count=1,
        )

        # Store the context item and update the priority bucket
        self.context_items[context_id] = context_item
        self.priority_buckets[initial_priority].append(context_id)

        return context_id

    def prioritize_context(
        self, query: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Prioritize context items based on query relevance and other factors.

        Args:
            query: Query string for matching context
            top_k: Number of top items to return
            min_similarity: Minimum similarity threshold to consider

        Returns:
            List[Tuple[str, float]]: List of (context_id, final priority score) tuples
        """
        # Generate an embedding for the query
        query_embedding = self._generate_embedding(query)

        # List to hold computed priority scores
        priority_scores = []

        # Iterate through all context items to calculate their scores
        for context_id, item in self.context_items.items():
            # Calculate semantic similarity between query and context
            similarity = self._calculate_similarity(query_embedding, item.embedding)

            # Skip items that do not meet the minimum similarity
            if similarity < min_similarity:
                continue

            # Compute time decay based on the age of the context
            time_factor = self._calculate_time_decay(item)

            # Compute usage factor based on access patterns
            usage_factor = self._calculate_usage_factor(item)

            # Combine all factors with the base priority to get a final score
            final_score = self._combine_priority_factors(
                similarity=similarity,
                time_factor=time_factor,
                usage_factor=usage_factor,
                base_priority=item.priority_score,
            )

            priority_scores.append((context_id, final_score))

        # Return the top-k context items sorted by final priority score
        return sorted(priority_scores, key=lambda x: x[1], reverse=True)[:top_k]

    def update_priority(self, context_id: str, new_priority: PriorityLevel) -> None:
        """
        Update the priority level of a specific context item.

        Args:
            context_id: Unique identifier of the context item
            new_priority: New PriorityLevel to assign
        """
        if context_id not in self.context_items:
            raise KeyError(f"Context ID {context_id} not found")

        # Remove the context item from its current priority bucket
        old_priority = self._get_priority_level(
            self.context_items[context_id].priority_score
        )
        self.priority_buckets[old_priority].remove(context_id)

        # Update the priority score and add to the new bucket
        self.context_items[context_id].priority_score = new_priority.value
        self.priority_buckets[new_priority].append(context_id)

    def batch_update_priorities(self, updates: List[Tuple[str, PriorityLevel]]) -> None:
        """
        Batch update priorities for multiple context items.

        Args:
            updates: List of tuples (context_id, new_priority)
        """
        for context_id, new_priority in updates:
            self.update_priority(context_id, new_priority)

    def get_context(self, context_id: str) -> Tuple[Union[str, dict], float]:
        """
        Retrieve a context item and update its access metrics.

        Args:
            context_id: Unique identifier of the context item

        Returns:
            Tuple containing the context content and its current priority score
        """
        if context_id not in self.context_items:
            raise KeyError(f"Context ID {context_id} not found")

        item = self.context_items[context_id]

        # Update access statistics
        item.last_accessed = datetime.now()
        item.access_count += 1

        return item.content, item.priority_score

    def _generate_embedding(self, content: Union[str, dict]) -> np.ndarray:
        """
        Generate an embedding vector for the given content using the embedding model.

        Args:
            content: Content to embed (str or dict)

        Returns:
            np.ndarray: Generated embedding vector
        """
        # Convert dictionary content to a JSON string for consistency
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)

        # Use cache to avoid redundant embedding computations
        cache_key = hash(content_str)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Generate the embedding without tracking gradients
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                content_str, device=self.device, normalize_embeddings=True
            )

        # Cache the generated embedding for future use
        self.embedding_cache[cache_key] = embedding

        return embedding

    def _calculate_similarity(
        self, query_embedding: np.ndarray, context_embedding: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            query_embedding: Embedding vector for the query
            context_embedding: Embedding vector for the context item

        Returns:
            float: Cosine similarity score (ensured to be non-negative)
        """
        # Convert embeddings to tensors on the appropriate device
        query_tensor = torch.tensor(query_embedding).to(self.device)
        context_tensor = torch.tensor(context_embedding).to(self.device)

        # Compute cosine similarity
        similarity = cosine_similarity(
            query_tensor.unsqueeze(0), context_tensor.unsqueeze(0)
        ).item()

        return max(0.0, similarity)  # Ensure non-negative similarity

    def _calculate_time_decay(self, item: ContextItem) -> float:
        """
        Calculate a decay factor based on the age of the context item.

        Args:
            item: The context item

        Returns:
            float: Time decay factor (minimum value enforced at 0.1)
        """
        # Compute the age of the context in hours
        age_hours = (datetime.now() - item.creation_time).total_seconds() / 3600
        decay = np.exp(-self.base_decay_rate * age_hours)
        return max(0.1, decay)

    def _calculate_usage_factor(self, item: ContextItem) -> float:
        """
        Calculate a usage factor based on frequency and recency of access.

        Args:
            item: The context item

        Returns:
            float: Combined usage factor
        """
        # Normalize frequency (capped at 10 accesses)
        frequency = min(item.access_count / 10, 1.0)
        # Compute recency factor based on time since last accessed
        recency = np.exp(
            -0.1 * (datetime.now() - item.last_accessed).total_seconds() / 3600
        )

        # Combine frequency and recency with given weights
        return 0.4 * frequency + 0.6 * recency

    def _combine_priority_factors(
        self,
        similarity: float,
        time_factor: float,
        usage_factor: float,
        base_priority: float,
    ) -> float:
        """
        Combine semantic similarity, time decay, usage, and base priority into a final score.

        Args:
            similarity: Semantic similarity score
            time_factor: Time decay factor
            usage_factor: Usage factor
            base_priority: Original priority score

        Returns:
            float: Final combined priority score
        """
        # Define weights for each contributing factor
        weights = {"similarity": 0.4, "time": 0.2, "usage": 0.2, "base": 0.2}

        # Normalize base priority relative to maximum possible value
        normalized_base = base_priority / max(p.value for p in PriorityLevel)

        # Compute final score as weighted sum
        final_score = (
            weights["similarity"] * similarity
            + weights["time"] * time_factor
            + weights["usage"] * usage_factor
            + weights["base"] * normalized_base
        )

        return final_score

    def _evict_lowest_priority(self) -> None:
        """
        Evict the context item with the lowest priority score to free up space.
        """
        # Identify the lowest priority item based on score and last accessed time
        lowest_priority = min(
            (item for item in self.context_items.values()),
            key=lambda x: (x.priority_score, x.last_accessed),
        )

        # Find the context ID corresponding to the lowest priority item
        context_id = next(
            k for k, v in self.context_items.items() if v == lowest_priority
        )

        # Remove the item from storage and its corresponding priority bucket
        del self.context_items[context_id]
        priority_level = self._get_priority_level(lowest_priority.priority_score)
        self.priority_buckets[priority_level].remove(context_id)

    def _get_priority_level(self, score: float) -> PriorityLevel:
        """
        Map a numeric priority score to a PriorityLevel enum.

        Args:
            score: Numeric priority score

        Returns:
            PriorityLevel: Corresponding priority level
        """
        for level in PriorityLevel:
            if score >= level.value:
                return level
        return PriorityLevel.ARCHIVAL

    def _generate_context_id(self, content: Union[str, dict]) -> str:
        """
        Generate a unique identifier for a context item based on its content and the current timestamp.

        Args:
            content: The context content

        Returns:
            str: Unique context identifier
        """
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        content_hash = hash(content_str)
        return f"ctx_{content_hash}_{timestamp}"

    def get_priority_distribution(self) -> Dict[PriorityLevel, int]:
        """
        Get the distribution of context items across different priority levels.

        Returns:
            Dict[PriorityLevel, int]: Mapping of PriorityLevel to count of items
        """
        distribution = defaultdict(int)
        for item in self.context_items.values():
            level = self._get_priority_level(item.priority_score)
            distribution[level] += 1
        return dict(distribution)

    def clear_cache(self) -> None:
        """
        Clear the embedding cache to free up memory.
        """
        self.embedding_cache.clear()
