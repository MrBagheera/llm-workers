from dataclasses import dataclass
from typing import Dict, List
from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class SimpleTokenUsageTracker:
    """Tracks token usage for a single model using dataclass for performance."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0

    def update_from_message(self, message: AIMessage) -> None:
        """Extract and accumulate token usage from AIMessage response metadata."""
        # Try different metadata structures for provider compatibility
        usage = None

        if hasattr(message, 'usage_metadata'):
            usage = message.usage_metadata

        elif hasattr(message, 'response_metadata'):
            response_metadata = message.response_metadata

            # Modern LangChain format (Anthropic, OpenAI v2)
            if 'usage_metadata' in response_metadata:
                usage = response_metadata['usage_metadata']
            # Older format (OpenAI v1)
            elif 'token_usage' in response_metadata:
                usage = response_metadata['token_usage']
            # Direct usage in response metadata
            elif any(key in response_metadata for key in ['total_tokens', 'input_tokens', 'output_tokens']):
                usage = response_metadata

        if not usage:
            return

        # Accumulate tokens
        if 'total_tokens' in usage:
            self.total_tokens += usage['total_tokens']
        if 'input_tokens' in usage:
            self.input_tokens += usage['input_tokens']
        if 'output_tokens' in usage:
            self.output_tokens += usage['output_tokens']

        # Handle reasoning tokens (for models that support it)
        if 'output_token_details' in usage:
            details = usage['output_token_details']
            if 'reasoning' in details:
                self.reasoning_tokens += details['reasoning']

        # Handle cache tokens
        if 'input_token_details' in usage:
            details = usage['input_token_details']
            if 'cache_read' in details:
                self.cache_read_tokens += details['cache_read']

    def reset(self) -> None:
        """Reset all token counters."""
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cache_read_tokens = 0

    def format_usage(self) -> str | None:
        """Format usage for display. Returns None if no tokens are used."""
        if self.total_tokens == 0:
            return None

        parts = [f"{self.total_tokens:,}"]

        if self.input_tokens > 0 or self.output_tokens > 0:
            detail_parts = [f"{self.input_tokens:,} in", f"{self.output_tokens:,} out"]
            if self.reasoning_tokens > 0:
                detail_parts.append(f"{self.reasoning_tokens:,} reasoning")
            parts.append(f"({', '.join(detail_parts)})")

        if self.cache_read_tokens > 0:
            parts.append(f"| Cache: {self.cache_read_tokens:,}")

        return " ".join(parts)


class CompositeTokenUsageTracker:
    """Manages token usage tracking across multiple models.

    Tracks both:
    - Total usage (lifetime, per-model) - never reset, accumulates forever
    - Current usage (session, all models combined) - reset by /rewind
    """

    def __init__(self):
        # Total usage (lifetime): per-model tracking, never reset
        self._total_per_model: Dict[str, SimpleTokenUsageTracker] = {}

        # Current usage (session): all models combined, reset by rewind
        self._current_session = SimpleTokenUsageTracker()

    def update_from_message(self, message: AIMessage, model_name: str) -> None:
        """Update both total (per-model) and current (session) usage from AIMessage response metadata."""
        # Initialize model tracker if needed
        if model_name not in self._total_per_model:
            self._total_per_model[model_name] = SimpleTokenUsageTracker()

        # Update total usage for this specific model (never reset)
        self._total_per_model[model_name].update_from_message(message)

        # Update current session usage (all models combined, reset by rewind)
        self._current_session.update_from_message(message)

    def format_current_usage(self) -> str | None:
        """Format current session usage for display during conversation. Returns None if no tokens."""
        usage = self._current_session.format_usage()
        if usage is None:
            return None

        return f"Tokens: {usage}"

    def format_total_usage(self) -> str | None:
        """Format detailed per-model total (lifetime) usage for exit display. Returns None if no tokens."""
        total_tokens = sum(tracker.total_tokens for tracker in self._total_per_model.values())
        if total_tokens == 0:
            return None

        lines = [f"Total Session Tokens: {total_tokens:,} total"]

        # Show per-model breakdown of total usage
        for model_name, tracker in sorted(self._total_per_model.items()):
            if tracker.total_tokens > 0:
                detail_parts = [f"{tracker.input_tokens:,} in", f"{tracker.output_tokens:,} out"]
                if tracker.reasoning_tokens > 0:
                    detail_parts.append(f"{tracker.reasoning_tokens:,} reasoning")

                model_line = f"  {model_name}: {tracker.total_tokens:,} ({', '.join(detail_parts)})"

                if tracker.cache_read_tokens > 0:
                    model_line += f" | Cache: {tracker.cache_read_tokens:,}"

                lines.append(model_line)

        return '\n'.join(lines)

    def reset_current_usage(self, messages: List[BaseMessage], current_model: str) -> None:
        """Reset current session usage and recalculate from remaining messages.

        Note: Total (lifetime) usage per model is never reset.
        """
        # Reset only current session usage (not total per-model usage)
        self._current_session.reset()

        # Recalculate current session from remaining messages
        # Note: We assume current model for all messages since we don't store model info with messages
        # This is a reasonable limitation for rewind functionality
        for message in messages:
            if isinstance(message, AIMessage):
                self._current_session.update_from_message(message)