from typing import Dict, List, Optional, Union, Any
import asyncio
import json
from datetime import datetime
from enum import Enum
import hashlib
import logging
from dataclasses import dataclass
import uuid


# Define possible states for the context handoff process
class HandoffState(Enum):
    INITIATED = "initiated"
    VALIDATING = "validating"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"


# Define types of context handoff
class HandoffType(Enum):
    FULL = "full"  # Complete context transfer
    PARTIAL = "partial"  # Selected context transfer
    REFERENCE = "reference"  # Transfer context reference only


# Data class representing a packet of context data
@dataclass
class ContextPacket:
    context_id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    checksum: str
    timestamp: str
    source_agent: str
    target_agent: str
    handoff_type: HandoffType
    sequence_number: int = 0
    total_packets: int = 1


# Base exception for handoff-related errors
class HandoffError(Exception):
    """Base class for handoff-related errors."""

    pass


# Exception raised when context validation fails
class ValidationError(HandoffError):
    """Raised when context validation fails."""

    pass


# Exception raised when context transfer fails
class TransferError(HandoffError):
    """Raised when context transfer fails."""

    pass


# Class managing shared context between agents with thread-safe operations
class SharedContextPool:
    """Manages shared context between agents with thread-safe operations."""

    def __init__(self):
        # Dictionary to store context data by context_id
        self._context_store: Dict[str, Dict] = {}
        # Asyncio lock to ensure thread-safe operations
        self._lock = asyncio.Lock()
        # List to log access operations for auditing
        self._access_log: List[Dict] = []
        # Logger for logging events
        self._logger = logging.getLogger(__name__)

    async def store_context(self, context_id: str, context_data: Dict) -> None:
        """Store context data in the shared pool."""
        async with self._lock:
            self._context_store[context_id] = context_data
            # Log the store operation
            self._log_access("store", context_id)

    async def retrieve_context(self, context_id: str) -> Optional[Dict]:
        """Retrieve context data from the shared pool."""
        async with self._lock:
            context = self._context_store.get(context_id)
            if context:
                # Log the retrieve operation if context exists
                self._log_access("retrieve", context_id)
            return context

    async def transfer(
        self, source_agent: str, target_agent: str, context_data: Dict
    ) -> str:
        """
        Transfer context data between agents.

        Args:
            source_agent (str): ID of the source agent.
            target_agent (str): ID of the target agent.
            context_data (Dict): The context data to transfer.

        Returns:
            str: Generated context_id for the transferred context.
        """
        # Generate a unique context ID
        context_id = str(uuid.uuid4())
        async with self._lock:
            # Store the context along with metadata
            await self.store_context(
                context_id,
                {
                    "data": context_data,
                    "source": source_agent,
                    "target": target_agent,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return context_id

    def _log_access(self, operation: str, context_id: str) -> None:
        """Log context access operations for auditing purposes."""
        self._access_log.append(
            {
                "operation": operation,
                "context_id": context_id,
                "timestamp": datetime.now().isoformat(),
            }
        )


# Class managing secure context handoff between AI agents
class ContextHandoffProtocol:
    """
    Manages secure context handoff between AI agents with validation and synchronization.

    Features:
    - Secure context serialization and validation.
    - Atomic context transfers.
    - Support for different handoff types.
    - Error handling and recovery.
    - Access logging and monitoring.
    """

    def __init__(self):
        # Shared context pool instance
        self.context_pool = SharedContextPool()
        # Dictionary to track active handoffs by handoff_id
        self._active_handoffs: Dict[str, HandoffState] = {}
        # Logger for protocol events
        self._logger = logging.getLogger(__name__)
        # Buffer to hold context packets during handoff
        self._packet_buffer: Dict[str, List[ContextPacket]] = {}

    async def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        context_data: Dict[str, Any],
        handoff_type: HandoffType = HandoffType.FULL,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Initiate context handoff between agents.

        Args:
            source_agent (str): ID of the source agent.
            target_agent (str): ID of the target agent.
            context_data (Dict[str, Any]): Context data to be transferred.
            handoff_type (HandoffType): Type of handoff (default is FULL).
            metadata (Optional[Dict]): Additional metadata for the handoff.

        Returns:
            str: Generated handoff ID.
        """
        # Generate a unique handoff ID
        handoff_id = self._generate_handoff_id(source_agent, target_agent)

        try:
            # Prepare context packets for the handoff
            packets = self._prepare_context_packets(
                handoff_id,
                source_agent,
                target_agent,
                context_data,
                handoff_type,
                metadata or {},
            )

            # Store the packets in the buffer
            self._packet_buffer[handoff_id] = packets

            # Mark the handoff as initiated
            self._active_handoffs[handoff_id] = HandoffState.INITIATED

            # Log the handoff initiation
            self._logger.info(f"Handoff initiated: {handoff_id}")

            return handoff_id

        except Exception as e:
            self._logger.error(f"Handoff initiation failed: {str(e)}")
            raise HandoffError(f"Failed to initiate handoff: {str(e)}")

    async def process_handoff(self, handoff_id: str) -> bool:
        """
        Process and complete the context handoff.

        Args:
            handoff_id (str): Handoff identifier.

        Returns:
            bool: True if handoff completed successfully.
        """
        try:
            # Ensure the handoff ID is valid
            if handoff_id not in self._active_handoffs:
                raise HandoffError(f"Invalid handoff ID: {handoff_id}")

            # Update state to validating
            self._active_handoffs[handoff_id] = HandoffState.VALIDATING

            # Retrieve packets associated with the handoff
            packets = self._packet_buffer.get(handoff_id, [])
            if not packets:
                raise HandoffError(f"No packets found for handoff: {handoff_id}")

            # Validate packet integrity and consistency
            await self._validate_packets(packets)

            # Update state to transferring
            self._active_handoffs[handoff_id] = HandoffState.TRANSFERRING

            # Process packets to reconstruct context data
            context_data = await self._process_packets(packets)

            # Transfer the context data to the shared context pool
            first_packet = packets[0]
            await self.context_pool.transfer(
                first_packet.source_agent, first_packet.target_agent, context_data
            )

            # Mark the handoff as completed
            self._active_handoffs[handoff_id] = HandoffState.COMPLETED

            # Clean up the packet buffer
            del self._packet_buffer[handoff_id]

            return True

        except Exception as e:
            # Update handoff state to failed upon error
            self._active_handoffs[handoff_id] = HandoffState.FAILED
            self._logger.error(f"Handoff processing failed: {str(e)}")
            raise HandoffError(f"Failed to process handoff: {str(e)}")

    async def get_handoff_state(self, handoff_id: str) -> HandoffState:
        """
        Get the current state of a handoff.

        Args:
            handoff_id (str): Handoff identifier.

        Returns:
            HandoffState: Current state of the handoff.
        """
        return self._active_handoffs.get(handoff_id, HandoffState.FAILED)

    def _prepare_context_packets(
        self,
        handoff_id: str,
        source_agent: str,
        target_agent: str,
        context_data: Dict[str, Any],
        handoff_type: HandoffType,
        metadata: Dict,
    ) -> List[ContextPacket]:
        """
        Prepare context data for transfer by splitting it into packets.

        Args:
            handoff_id (str): Unique handoff identifier.
            source_agent (str): ID of the source agent.
            target_agent (str): ID of the target agent.
            context_data (Dict[str, Any]): Context data to be transferred.
            handoff_type (HandoffType): Type of handoff.
            metadata (Dict): Additional metadata.

        Returns:
            List[ContextPacket]: List containing the prepared context packet(s).
        """
        # Serialize the context data into JSON format
        serialized_data = json.dumps(context_data)

        # Calculate a checksum for data integrity verification
        checksum = hashlib.sha256(serialized_data.encode()).hexdigest()

        # Create a context packet (extendable for multiple packets if needed)
        packet = ContextPacket(
            context_id=handoff_id,
            content=context_data,
            metadata=metadata,
            checksum=checksum,
            timestamp=datetime.now().isoformat(),
            source_agent=source_agent,
            target_agent=target_agent,
            handoff_type=handoff_type,
        )

        return [packet]

    async def _validate_packets(self, packets: List[ContextPacket]) -> None:
        """
        Validate the integrity and consistency of context packets.

        Args:
            packets (List[ContextPacket]): List of context packets to validate.

        Raises:
            ValidationError: If any packet fails validation.
        """
        if not packets:
            raise ValidationError("No packets to validate")

        # Set to track received checksums
        received_checksums = set()
        for packet in packets:
            # Calculate the checksum for the packet content
            calculated_checksum = hashlib.sha256(
                json.dumps(packet.content).encode()
            ).hexdigest()

            if calculated_checksum != packet.checksum:
                raise ValidationError(
                    f"Checksum mismatch for packet: {packet.context_id}"
                )

            received_checksums.add(packet.checksum)

        # Perform additional validation checks
        await self._additional_validation(packets)

    async def _additional_validation(self, packets: List[ContextPacket]) -> None:
        """
        Perform additional validation checks on the packets.

        Args:
            packets (List[ContextPacket]): List of context packets.

        Raises:
            ValidationError: If any packet fails the additional validation.
        """
        # Check that packet timestamps are within acceptable limits
        current_time = datetime.now()
        for packet in packets:
            packet_time = datetime.fromisoformat(packet.timestamp)
            time_diff = (current_time - packet_time).total_seconds()

            if time_diff > 300:  # 5 minutes threshold
                raise ValidationError(f"Packet too old: {packet.context_id}")

    async def _process_packets(self, packets: List[ContextPacket]) -> Dict:
        """
        Process and combine context packets into a single context data dictionary.

        Args:
            packets (List[ContextPacket]): List of context packets.

        Returns:
            Dict: Combined context data.

        Raises:
            TransferError: If no packets are available for processing.
        """
        if not packets:
            raise TransferError("No packets to process")

        # For simplicity, return the content of the first packet
        return packets[0].content

    def _generate_handoff_id(self, source_agent: str, target_agent: str) -> str:
        """
        Generate a unique handoff identifier.

        Args:
            source_agent (str): ID of the source agent.
            target_agent (str): ID of the target agent.

        Returns:
            str: Unique handoff identifier.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_str = f"{source_agent}_{target_agent}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()


# Example usage of the ContextHandoffProtocol


def main():
    async def complex_handoff_example():
        protocol = ContextHandoffProtocol()

        # Prepare metadata for the handoff
        metadata = {
            "priority": "high",
            "retention_period": "24h",
            "security_level": "confidential",
        }

        try:
            # Initiate a partial context transfer
            handoff_id = await protocol.initiate_handoff(
                source_agent="agent_1",
                target_agent="agent_2",
                context_data={"selected_history": [...], "relevant_state": {...}},
                handoff_type=HandoffType.PARTIAL,
                metadata=metadata,
            )

            # Monitor the handoff state until it is completed or failed
            while True:
                state = await protocol.get_handoff_state(handoff_id)
                if state in [HandoffState.COMPLETED, HandoffState.FAILED]:
                    break
                await asyncio.sleep(0.1)

            if state == HandoffState.COMPLETED:
                # Retrieve transferred context if handoff succeeded
                context = await protocol.context_pool.retrieve_context(handoff_id)
                # Further processing of the context can be done here
                protocol._logger.info(
                    f"Handoff completed successfully for ID: {handoff_id}"
                )

        except HandoffError as e:
            # Log and handle handoff error
            protocol._logger.error(f"Handoff error occurred: {e}")

    # Run the asynchronous handoff example
    asyncio.run(complex_handoff_example())


if __name__ == "__main__":
    main()
