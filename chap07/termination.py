"""
Module: Termination Detection
This module contains a function to detect if a conversation should be gracefully terminated based on user messages.
"""


def detect_termination(user_message):
    """
    Detects if a conversation should be gracefully terminated.

    This function checks the user_message for any termination phrases such as "thank you", "bye",
    "that's all", or "goodbye". If any of these phrases are found, it returns True indicating
    that the conversation should be terminated.

    Args:
        user_message (str): The message input by the user.

    Returns:
        bool: True if a termination phrase is detected, otherwise False.
    """
    # Define a list of termination phrases to check in the message.
    termination_phrases = ["thank you", "bye", "that's all", "goodbye"]

    # Convert the message to lower case and check for each termination phrase.
    for phrase in termination_phrases:
        if phrase in user_message.lower():
            return True
    return False


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Sample user input for termination detection.
    user_input = "Thanks, thats all I needed."

    # Evaluate whether the conversation should be terminated.
    should_terminate = detect_termination(user_input)

    # Output the termination evaluation result.
    print(f"Should terminate: {should_terminate}")
