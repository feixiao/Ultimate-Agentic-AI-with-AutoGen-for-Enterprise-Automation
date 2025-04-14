from typing import Dict, List, Any
import tiktoken
import json
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum


# Enum to represent different context priority levels
class ContextPriority(Enum):
    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


# Dataclass to hold context template details and metadata
@dataclass
class ContextTemplate:
    name: str  # Unique template name
    template: str  # Template string with placeholders
    priority: ContextPriority  # Priority level for the template
    max_length: Optional[int] = None  # Optional maximum length for the template content
    required_fields: List[str] = (
        None  # List of required fields extracted from the template
    )

    def __post_init__(self):
        # If no required fields are provided, extract placeholders from the template
        if self.required_fields is None:
            self.required_fields = re.findall(r"{(\w+)}", self.template)


# Class for dynamic context injection into prompts
class DynamicContextInjector:
    """
    Dynamically injects context into prompts based on templates and priorities.

    Handles formatting, prioritization, and injection of various context types
    while managing token limits and maintaining relevance.
    """

    def __init__(self, max_total_length: int = 2048):
        # Maximum allowed length of the final prompt
        self.max_total_length = max_total_length
        # Initialize default context templates
        self.templates = self._initialize_default_templates()
        # Store the history of context injections
        self.context_history = []

    def _initialize_default_templates(self) -> Dict[str, ContextTemplate]:
        """
        Initialize default context templates with pre-defined priorities.

        Returns:
            Dict[str, ContextTemplate]: Mapping of template names to ContextTemplate objects.
        """
        return {
            "system_config": ContextTemplate(
                name="system_config",
                template="System Configuration: {config}",
                priority=ContextPriority.CRITICAL,
                max_length=200,
            ),
            "user_preference": ContextTemplate(
                name="user_preference",
                template="User Preferences: {preferences}",
                priority=ContextPriority.HIGH,
                max_length=300,
            ),
            "conversation_history": ContextTemplate(
                name="conversation_history",
                template="Previous Interactions:\n{history}",
                priority=ContextPriority.MEDIUM,
                max_length=500,
            ),
            "system_state": ContextTemplate(
                name="system_state",
                template="Current System State: {state}",
                priority=ContextPriority.HIGH,
                max_length=200,
            ),
            "metadata": ContextTemplate(
                name="metadata",
                template="Session Metadata: {metadata}",
                priority=ContextPriority.LOW,
                max_length=100,
            ),
        }

    def add_template(
        self,
        name: str,
        template: str,
        priority: ContextPriority,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Add or update a context template.

        Args:
            name (str): Unique identifier for the template.
            template (str): String template with placeholders.
            priority (ContextPriority): Priority level for this context.
            max_length (Optional[int]): Maximum length for this context type.
        """
        self.templates[name] = ContextTemplate(
            name=name, template=template, priority=priority, max_length=max_length
        )

    def inject_context(
        self,
        prompt: str,
        context_data: Dict[str, dict],
        max_length: Optional[int] = None,
    ) -> str:
        """
        Inject context into the prompt based on available templates and priorities.

        Args:
            prompt (str): Base prompt to inject context into.
            context_data (Dict[str, dict]): Context data keyed by template name.
            max_length (Optional[int]): Optional override for max total length.

        Returns:
            str: The formatted prompt with injected context.
        """
        max_len = max_length or self.max_total_length
        formatted_contexts = []

        # Sort templates by descending priority value
        sorted_templates = sorted(
            self.templates.items(), key=lambda x: x[1].priority.value, reverse=True
        )

        # Start with the length of the base prompt
        current_length = len(prompt)

        # Process each template in order of priority
        for template_name, template in sorted_templates:
            # Skip if no context data provided for the template
            if template_name not in context_data:
                continue

            context = context_data[template_name]

            # Validate that all required fields are present in the context data
            if not self._validate_context_data(template, context):
                continue

            # Format the context data using the template
            formatted_context = self._format_context(template, context)

            # Check if adding this formatted context exceeds the maximum allowed length
            if current_length + len(formatted_context) > max_len:
                # Attempt to compress the context to fit the remaining length
                compressed_context = self._compress_context(
                    formatted_context, max_len - current_length
                )
                if compressed_context:
                    formatted_contexts.append(compressed_context)
                    current_length += len(compressed_context)
            else:
                formatted_contexts.append(formatted_context)
                current_length += len(formatted_context)

        # Combine the formatted contexts with the base prompt
        result = "\n".join(formatted_contexts + [prompt])

        # Update context history with the current injection
        self._update_history(prompt, context_data)

        return result

    def _validate_context_data(self, template: ContextTemplate, context: dict) -> bool:
        """
        Validate that the provided context data contains all required fields.

        Args:
            template (ContextTemplate): The context template.
            context (dict): The context data.

        Returns:
            bool: True if all required fields are present, False otherwise.
        """
        return all(field in context for field in template.required_fields)

    def _format_context(self, template: ContextTemplate, context: dict) -> str:
        """
        Format the context data using the specified template, applying length limits.

        Args:
            template (ContextTemplate): The template to use.
            context (dict): The context data.

        Returns:
            str: The formatted context string.
        """
        formatted_values = {}
        for field in template.required_fields:
            value = context[field]
            # Convert complex types (dict, list) to JSON string for consistency
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            # Truncate the value if it exceeds the maximum length defined in the template
            formatted_values[field] = self._truncate_text(
                str(value), template.max_length
            )

        return template.template.format(**formatted_values)

    def _compress_context(self, context: str, max_length: int) -> Optional[str]:
        """
        Attempt to compress context to fit within a specified length.

        Uses strategies such as whitespace normalization and truncation.

        Args:
            context (str): The context string to compress.
            max_length (int): The maximum allowed length.

        Returns:
            Optional[str]: Compressed context string if possible, otherwise None.
        """
        if len(context) <= max_length:
            return context

        # Normalize whitespace and punctuation
        compressed = re.sub(r"\s+", " ", context).strip()
        compressed = re.sub(r"\s*([,.])\s*", r"\1 ", compressed)

        if len(compressed) <= max_length:
            return compressed

        # Truncate and add ellipsis if still too long
        return compressed[: max_length - 3] + "..."

    def _truncate_text(self, text: str, max_length: Optional[int]) -> str:
        """
        Truncate text to a specified maximum length, adding ellipsis if truncated.

        Args:
            text (str): The text to truncate.
            max_length (Optional[int]): Maximum allowed length.

        Returns:
            str: The truncated text.
        """
        if max_length and len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    def _update_history(self, prompt: str, context_data: Dict[str, dict]) -> None:
        """
        Update the context history with the current prompt and context data.

        Args:
            prompt (str): The prompt used.
            context_data (Dict[str, dict]): The context data used.
        """
        self.context_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "context_data": context_data,
            }
        )

        # Limit history size to 1000 entries
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]

    def get_context_history(self, limit: Optional[int] = None) -> List[dict]:
        """
        Retrieve recent context history.

        Args:
            limit (Optional[int]): Maximum number of history entries to return.

        Returns:
            List[dict]: List of context history entries.
        """
        if limit:
            return self.context_history[-limit:]
        return self.context_history

    def clear_history(self) -> None:
        """
        Clear all stored context history.
        """
        self.context_history = []


# Usage example
def main():
    # Initialize the DynamicContextInjector with a maximum total length of 4096 tokens
    injector = DynamicContextInjector(max_total_length=4096)

    # Add a custom template if needed
    injector.add_template(
        name="user_goal",
        template="Current User Goal: {goal_description}",
        priority=ContextPriority.HIGH,
        max_length=200,
    )

    # Prepare context data with multiple context types
    context_data = {
        "system_config": {"config": {"model": "gpt-4", "temperature": 0.7}},
        "user_preference": {
            "preferences": {"language": "English", "expertise_level": "Expert"}
        },
        "conversation_history": {
            "history": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ]
        },
        "user_goal": {"goal_description": "Analyzing system performance metrics"},
    }

    # Define a base prompt for the agent
    prompt = "What are the current system metrics?"

    # Inject context into the prompt using the defined templates and context data
    result = injector.inject_context(prompt, context_data)

    # Output the final prompt with the injected context
    print(result)


if __name__ == "__main__":
    main()
