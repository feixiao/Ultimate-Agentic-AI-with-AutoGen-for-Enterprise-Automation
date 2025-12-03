import re
import math
from collections import Counter


def detect_prompt_injection(user_input, conversation_history):
    """
    Detect potential prompt injection attacks in user input.

    Args:
        user_input (str): The text input from the user.
        conversation_history (list): List of previous conversation messages.

    Returns:
        dict: Analysis results with detection status, type, risk level, and evidence.
    """
    # 1. Check for explicit instruction override patterns
    instruction_patterns = [
        r"ignore (all |)previous instructions",
        r"disregard (all |)instructions",
        r"system\s*:\s*|<system>",
        r"you (are now|will act as) ",
        r"as an AI (without|lacking) restrictions",
    ]
    for pattern in instruction_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return {
                "detected": True,
                "type": "explicit_instruction_override",
                "risk": "high",
                "evidence": match.group(0),
            }

    # 2. Semantic shift detection using conversation context
    if conversation_history:
        topic_shift = detect_topic_shift(user_input, conversation_history)
        if topic_shift["is_significant"] and contains_system_terms(user_input):
            return {
                "detected": True,
                "type": "multi_turn_attack",
                "risk": "medium",
                "evidence": "Significant topic shift with system terminology",
            }

    # 3. Delimiter and formatting analysis for suspicious patterns
    suspicious_formatting = [
        r"<.*>",  # XML-like tags
        r"```.*```",  # Code blocks that might include instructions
        r"^system:",  # Lines starting with 'system:'
        r"\[.*\]\{.*\}",  # Special formatting using brackets
    ]
    for pattern in suspicious_formatting:
        if re.search(pattern, user_input, re.IGNORECASE | re.DOTALL):
            return {
                "detected": True,
                "type": "delimiter_confusion",
                "risk": "medium",
                "evidence": "Contains formatting that might confuse context boundaries",
            }

    # 4. Check for role-playing and persona override patterns
    persona_patterns = [
        r"you are (now |)([A-Za-z]+Mode)",
        r"assume the role of",
        r"act as (if you (are|were)|a) ",
        r"you're (a|an) unfiltered",
    ]
    for pattern in persona_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return {
                "detected": True,
                "type": "role_playing_attack",
                "risk": "high",
                "evidence": match.group(0),
            }

    # 5. Evaluate text entropy and pattern complexity to detect obfuscation
    entropy = calculate_entropy(user_input)
    unusual_pattern_score = detect_unusual_patterns(user_input)
    if entropy > 5.2 and unusual_pattern_score > 0.7:
        return {
            "detected": True,
            "type": "obfuscated_injection",
            "risk": "medium",
            "evidence": f"High entropy ({entropy:.2f}) with unusual patterns",
        }

    # No injection detected; return low risk result.
    return {"detected": False, "risk": "low"}


def contains_system_terms(text):
    """
    Check if text contains terms commonly used in system prompts.

    Args:
        text (str): Text to analyze.

    Returns:
        bool: True if system-related terms are found, else False.
    """
    system_terms = [
        "instruction",
        "system",
        "prompt",
        "ai",
        "model",
        "override",
        "command",
        "token",
        "parameter",
        "reset",
    ]
    for term in system_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
            return True
    return False


def detect_topic_shift(current_input, history):
    """
    Detect significant shifts in conversation topics.

    Args:
        current_input (str): The current user input.
        history (list): The conversation history.

    Returns:
        dict: Contains 'is_significant' flag and 'shift_score' value.
    """
    # Simplified implementation using basic logic.
    # In production, use embeddings and cosine similarity for accurate detection.
    return {"is_significant": False, "shift_score": 0.3}


def calculate_entropy(text):
    """
    Calculate the Shannon entropy of a text string.

    Args:
        text (str): Text input for entropy calculation.

    Returns:
        float: Entropy value; higher value indicates more randomness.
    """
    if not text:
        return 0

    # Count frequency of each character in the text
    char_counts = Counter(text)
    length = len(text)
    entropy = 0

    # Calculate entropy using probability of each character
    for count in char_counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def detect_unusual_patterns(text):
    """
    Analyze text for unusual patterns that may indicate obfuscation.

    Args:
        text (str): The text to analyze.

    Returns:
        float: A score representing the likelihood of unusual patterns.
    """
    score = 0.0

    # Check for unicode homoglyphs
    homoglyph_count = count_potential_homoglyphs(text)
    if homoglyph_count > 0:
        score += 0.3

    # Evaluate ratio of special characters in text
    special_char_ratio = len(
        [c for c in text if not c.isalnum() and not c.isspace()]
    ) / max(len(text), 1)
    if special_char_ratio > 0.3:
        score += 0.2

    # Check for nested delimiters
    if has_nested_delimiters(text):
        score += 0.2

    # Check for reversed text segments
    if has_reversed_text(text):
        score += 0.3

    return score


def count_potential_homoglyphs(text):
    """
    Count potential unicode homoglyphs in text.

    Args:
        text (str): The text to analyze.

    Returns:
        int: Number of characters in suspicious unicode ranges.
    """
    # Define ranges for potential homoglyphs (Cyrillic, Greek, Technical)
    suspicious_ranges = [
        (0x0400, 0x04FF),  # Cyrillic range
        (0x0370, 0x03FF),  # Greek range
        (0x2000, 0x23FF),  # Technical, arrows, math symbols
    ]
    count = 0
    for char in text:
        code_point = ord(char)
        for start, end in suspicious_ranges:
            if start <= code_point <= end:
                count += 1
                break
    return count


def has_nested_delimiters(text):
    """
    Check for suspicious nesting of delimiters like quotes or brackets.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if nested delimiters detected, otherwise False.
    """
    stack = []
    pairs = {")": "(", "}": "{", "]": "[", '"': '"', "'": "'"}
    nesting_level = 0
    max_nesting = 0

    for char in text:
        if char in "({[\"'":
            stack.append(char)
            nesting_level += 1
            max_nesting = max(max_nesting, nesting_level)
        elif char in ")}]\"'":
            if not stack or stack[-1] != pairs.get(char, None):
                # Mismatched delimiter indicates suspicious nesting
                return True
            stack.pop()
            nesting_level -= 1

    # Consider deep nesting (beyond level 3) as suspicious
    return max_nesting > 3


def has_reversed_text(text):
    """
    Check for reversed text segments that may be used to hide instructions.

    Args:
        text (str): The text to analyze.

    Returns:
        bool: True if reversed text is detected, otherwise False.
    """
    words = text.split()
    for word in words:
        # If word is long and equals its reverse, flag as reversed
        if len(word) > 4 and word == word[::-1]:
            return True
    return False
