"""
Module: Multi-Agent Evaluator
This module defines a MultiAgentEvaluator class to log interactions between agents and evaluate message consistency.
"""


class MultiAgentEvaluator:
    def __init__(self):
        # Initialize a dictionary to store logs for each receiver agent.
        self.agent_logs = {}

    def log_interaction(self, sender, receiver, message):
        """
        Logs an interaction between two agents.

        Args:
            sender (str): The agent sending the message.
            receiver (str): The agent receiving the message.
            message (str): The content of the message.
        """
        # If the receiver doesn't have a log yet, initialize an empty list.
        if receiver not in self.agent_logs:
            self.agent_logs[receiver] = []
        # Append the interaction as a dictionary entry.
        self.agent_logs[receiver].append({"from": sender, "message": message})

    def evaluate_message_consistency(self, expected_messages):
        """
        Evaluates if the logged messages for each agent match the expected messages.

        Args:
            expected_messages (dict): A dictionary with agent names as keys and lists of expected messages as values.

        Returns:
            dict: A dictionary containing, for each agent:
                - 'matched': True if actual messages equal expected messages,
                - 'received': List of messages received,
                - 'expected': The expected messages list.
        """
        consistency_results = {}
        for agent, received_messages in self.agent_logs.items():
            # Extract the list of messages from the logs.
            actual_messages = [msg["message"] for msg in received_messages]
            consistency_results[agent] = {
                "matched": actual_messages == expected_messages.get(agent, []),
                "received": actual_messages,
                "expected": expected_messages.get(agent, []),
            }
        return consistency_results


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of MultiAgentEvaluator
    evaluator = MultiAgentEvaluator()

    # Log interactions between agents
    evaluator.log_interaction("Agent A", "Agent B", "Retrieve data from database")
    evaluator.log_interaction("Agent B", "Agent C", "Processing retrieved data")

    # Define the expected messages for each agent
    expected_msgs = {
        "Agent B": ["Retrieve data from database"],
        "Agent C": ["Processing retrieved data"],
    }

    # Evaluate and print the message consistency
    print(evaluator.evaluate_message_consistency(expected_msgs))
