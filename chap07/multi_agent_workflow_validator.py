"""
Module: Multi-Agent Workflow Validator
This module defines a WorkflowValidator class to log and validate the execution order of tasks by agents.
"""


class WorkflowValidator:
    def __init__(self, expected_sequence):
        """
        Initialize the WorkflowValidator with the expected sequence of tasks.

        Args:
            expected_sequence (list): List of tasks in the expected execution order.
        """
        self.expected_sequence = expected_sequence
        self.executed_tasks = []  # List to store tuples of (agent, task)

    def log_task_execution(self, agent, task):
        """
        Logs the execution of a task by an agent.

        Args:
            agent (str): The identifier of the agent executing the task.
            task (str): The task that was executed.
        """
        # Append the (agent, task) tuple to the executed tasks log
        self.executed_tasks.append((agent, task))

    def validate_execution_order(self):
        """
        Validates if the executed task order matches the expected sequence.

        Returns:
            dict: A dictionary containing:
                - 'valid_order': True if the executed sequence matches the expected sequence.
                - 'executed_sequence': The list of tasks as executed.
                - 'expected_sequence': The expected list of tasks.
        """
        # Extract the executed tasks in order
        executed_sequence = [task for _, task in self.executed_tasks]
        return {
            "valid_order": executed_sequence == self.expected_sequence,
            "executed_sequence": executed_sequence,
            "expected_sequence": self.expected_sequence,
        }


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the expected sequence of tasks
    expected_sequence = ["Fetch Data", "Process Data", "Generate Report"]

    # Initialize the WorkflowValidator with the expected sequence
    validator = WorkflowValidator(expected_sequence)

    # Log task executions by different agents
    validator.log_task_execution("Agent A", "Fetch Data")
    validator.log_task_execution("Agent B", "Process Data")
    validator.log_task_execution("Agent C", "Generate Report")

    # Validate the execution order and print the result
    print(validator.validate_execution_order())
