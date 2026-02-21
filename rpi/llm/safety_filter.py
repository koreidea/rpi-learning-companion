from dataclasses import dataclass
from typing import Optional

from loguru import logger

from core.config import ConfigManager


@dataclass
class FilterResult:
    blocked: bool = False
    redirect_response: str = ""


# Words/patterns that should trigger redirection in child's input
INPUT_BLOCKLIST = [
    "kill", "die", "dead", "murder", "blood", "gun", "weapon",
    "sex", "naked", "drug", "alcohol", "beer", "wine",
    "hate", "stupid", "idiot", "dumb",
    "password", "address", "phone number",
]

# Words that should not appear in LLM output
OUTPUT_BLOCKLIST = [
    "kill", "die", "murder", "blood", "weapon", "gun",
    "sex", "naked", "drug", "alcohol",
    "stupid", "idiot", "dumb", "ugly",
    "scary", "terrifying", "horror", "nightmare",
    "politics", "election", "president",
]

GENTLE_REDIRECTS = [
    "That's a great question! Let's talk about something fun instead. Do you want to hear about animals or space?",
    "Hmm, let's think about something exciting! Do you know what the biggest animal in the ocean is?",
    "I love your curiosity! Let's learn something cool. Did you know butterflies taste with their feet?",
    "Ooh, let me tell you something amazing instead! Do you know how rainbows are made?",
]


class SafetyFilter:
    """Content safety filter for child-appropriate interactions.

    Two-layer filtering:
    1. Input filter: checks child's speech for concerning content
    2. Output filter: checks LLM responses before TTS
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._redirect_index = 0

    def check_input(self, text: str) -> FilterResult:
        """Check the child's transcribed speech for concerning content."""
        lower = text.lower()

        for word in INPUT_BLOCKLIST:
            if word in lower:
                logger.warning("Input filter triggered on: '{}'", word)
                return FilterResult(
                    blocked=True,
                    redirect_response=self._get_redirect(),
                )

        return FilterResult(blocked=False)

    def check_output(self, text: str) -> FilterResult:
        """Check LLM output for inappropriate content before TTS."""
        lower = text.lower()

        for word in OUTPUT_BLOCKLIST:
            if word in lower:
                logger.warning("Output filter triggered on: '{}'", word)
                return FilterResult(
                    blocked=True,
                    redirect_response="Let me think of something better to say!",
                )

        # Check response length (for young children, keep it short)
        config = self.config_manager.config
        if config.child.age_max <= 6:
            # Count sentences
            sentence_count = text.count(".") + text.count("!") + text.count("?")
            if sentence_count > 4:
                # Truncate to ~3 sentences
                parts = []
                count = 0
                for char in text:
                    parts.append(char)
                    if char in ".!?":
                        count += 1
                        if count >= 3:
                            break
                text = "".join(parts)
                return FilterResult(blocked=True, redirect_response=text)

        return FilterResult(blocked=False)

    def _get_redirect(self) -> str:
        """Get the next gentle redirect response (cycles through options)."""
        response = GENTLE_REDIRECTS[self._redirect_index % len(GENTLE_REDIRECTS)]
        self._redirect_index += 1
        return response
