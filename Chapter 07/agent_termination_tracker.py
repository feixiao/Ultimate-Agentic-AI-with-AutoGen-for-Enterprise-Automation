"""
Module: Agent Termination Tracker
This module provides a class to track active agents and verify when they have completed their tasks.
"""


class TerminationTracker:
    def __init__(self):
        # Initialize a set to keep track of active agents
        self.active_agents = set()

    def start_task(self, agent):
        """
        Marks an agent as active by adding it to the active_agents set.

        Args:
            agent: The identifier for the agent to be tracked.
        """
        self.active_agents.add(agent)

    def complete_task(self, agent):
        """
        Marks an agent as completed by removing it from the active_agents set.

        Args:
            agent: The identifier for the agent that has completed its task.
        """
        if agent in self.active_agents:
            self.active_agents.remove(agent)

    def validate_termination(self):
        """
        Checks if all agents have completed their tasks.

        Returns:
            dict: A dictionary with:
                - 'all_agents_terminated': True if no active agents remain,
                - 'remaining_agents': List of agents still active.
        """
        return {
            "all_agents_terminated": len(self.active_agents) == 0,
            "remaining_agents": list(self.active_agents),
        }


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create an instance of TerminationTracker
    tracker = TerminationTracker()

    # Start tasks for agents
    tracker.start_task("Agent A")
    tracker.start_task("Agent B")

    # Mark tasks as completed
    tracker.complete_task("Agent A")
    tracker.complete_task("Agent B")

    # Validate termination status and print results
    status = tracker.validate_termination()
    print(status)
