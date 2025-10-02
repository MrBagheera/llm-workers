from dataclasses import dataclass
from typing import Dict, Any

from langchain_core.messages import BaseMessage


def _extract_usage_metadata_from_message(message: BaseMessage, default_model_name: str) -> Dict[str, Dict[str, Any]] | None:
    """Extract usage metadata dictionary from a message."""
    # Check additional_kwargs first (for tool-passed usage metadata)
    if hasattr(message, 'additional_kwargs') and 'usage_metadata_per_model' in message.additional_kwargs:
        return message.additional_kwargs['usage_metadata_per_model']

    elif hasattr(message, 'usage_metadata'):
        return {default_model_name: message.usage_metadata}

    elif hasattr(message, 'response_metadata'):
        response_metadata = message.response_metadata

        # Modern LangChain format (Anthropic, OpenAI v2)
        if 'usage_metadata' in response_metadata:
            return {default_model_name: response_metadata['usage_metadata']}
        # Older format (OpenAI v1)
        elif 'token_usage' in response_metadata:
            return {default_model_name: response_metadata['token_usage']}
        # Direct usage in response metadata
        elif any(key in response_metadata for key in ['total_tokens', 'input_tokens', 'output_tokens']):
            return {default_model_name: response_metadata}

    return None


@dataclass
class SimpleTokenUsageTracker:
    """Tracks token usage for a single model using dataclass for performance."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0

    def update_from_metadata(self, usage_metadata: Dict[str, Any]) -> None:
        """Update token counts from usage metadata dictionary."""
        if not usage_metadata:
            return

        # Accumulate tokens
        if 'total_tokens' in usage_metadata:
            self.total_tokens += usage_metadata['total_tokens']
        if 'input_tokens' in usage_metadata:
            self.input_tokens += usage_metadata['input_tokens']
        if 'output_tokens' in usage_metadata:
            self.output_tokens += usage_metadata['output_tokens']

        # Handle reasoning tokens (for models that support it)
        if 'output_token_details' in usage_metadata:
            details = usage_metadata['output_token_details']
            if 'reasoning' in details:
                self.reasoning_tokens += details['reasoning']

        # Handle cache tokens
        if 'input_token_details' in usage_metadata:
            details = usage_metadata['input_token_details']
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
    - Current usage (all models combined) - reset by each call to format_current_usage
    """

    def __init__(self):
        # Total usage (lifetime): per-model tracking, never reset
        self._total_per_model: Dict[str, SimpleTokenUsageTracker] = {}

        # Current usage (session): all models combined, reset by rewind
        self._current = SimpleTokenUsageTracker()

    def update_from_message(self, message: BaseMessage, model_name: str) -> None:
        """Update both total (per-model) and current (session) usage from BaseMessage metadata."""

        usage_metadata_per_model = _extract_usage_metadata_from_message(message, model_name)
        if not usage_metadata_per_model:
            return

        self.update_from_metadata(usage_metadata_per_model)

    def update_from_metadata(self, usage_metadata_per_model: Dict[str, Dict[str, Any]], update_only_current: bool = False) -> None:
        """Update both total (per-model) and current (session) usage from usage metadata."""
        for model_name, usage_metadata in usage_metadata_per_model.items():
            if not update_only_current:
                if model_name not in self._total_per_model:
                    self._total_per_model[model_name] = SimpleTokenUsageTracker()
                self._total_per_model[model_name].update_from_metadata(usage_metadata)

            self._current.update_from_metadata(usage_metadata)

    def attach_usage_to_message(self, message: BaseMessage) -> None:
        """Attach token usage metadata to a message via additional_kwargs."""
        usage_metadata = {}
        for model_name, tracker in self._total_per_model.items():
            if tracker.total_tokens > 0:
                usage_metadata[model_name] = {
                    'total_tokens': tracker.total_tokens,
                    'input_tokens': tracker.input_tokens,
                    'output_tokens': tracker.output_tokens
                }
                if tracker.reasoning_tokens > 0:
                    usage_metadata[model_name]['output_token_details'] = {'reasoning': tracker.reasoning_tokens}
                if tracker.cache_read_tokens > 0:
                    usage_metadata[model_name]['input_token_details'] = {'cache_read': tracker.cache_read_tokens}

        if not hasattr(message, 'additional_kwargs'):
            message.additional_kwargs = {}
        message.additional_kwargs['usage_metadata'] = usage_metadata

    def format_current_usage(self) -> str | None:
        """Format current session usage for display during conversation. Returns None if no tokens."""
        usage = self._current.format_usage()
        if usage is None:
            return None

        self._current.reset()
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