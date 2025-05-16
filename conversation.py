import yaml
from typing import List, Dict, Tuple
from pathlib import Path


class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict] = []
        self._load_prompts()

    def _load_prompts(self):
        prompt_path = Path(__file__).parent / "prompt.yaml"
        with open(prompt_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def add_message(
        self, role: str, content: str, context: str = None, package_name: str = None
    ) -> Tuple[str, List[Dict]]:
        """Add a message to the conversation history and format the prompt.

        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The message content
            context: Optional context for the message
            package_name: Optional package name for system prompt

        Returns:
            Tuple of (formatted_prompt, messages_list)
        """
        self.history.append({"role": role, "content": content})
        if (
            len(self.history) > self.max_history * 2
        ):  # *2 because each exchange has 2 messages
            self.history = self.history[-self.max_history * 2 :]

        # Always return messages, even if context/package_name not provided
        if context and package_name:
            # Format the conversation history
            history_str = self.format_history()

            # Format the main prompt
            prompt = self.prompts["conversation_template"].format(
                context=context, history=history_str, question=content
            )

            # Create the messages list for the API
            messages = [
                {"role": "system", "content": self.get_system_prompt(package_name)},
                *self.history,
                {"role": "user", "content": prompt},
            ]

            return prompt, messages
        else:
            # Return just the history without additional formatting
            return None, self.history

    def format_history(self) -> str:
        """Format the conversation history as a string."""
        formatted_history = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        return "\n".join(formatted_history)

    def get_system_prompt(self, package_name: str) -> str:
        """Get the system prompt with package name filled in."""
        return self.prompts["system_prompt"].format(package_name=package_name)
