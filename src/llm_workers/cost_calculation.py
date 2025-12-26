"""Cost calculation module for estimating LLM API costs based on token usage."""

from dataclasses import dataclass
from typing import Dict, Optional

from llm_workers.config import PricingConfig
from llm_workers.token_tracking import SimpleTokenUsageTracker


@dataclass
class ModelCost:
    """Represents calculated cost for a single model."""
    currency: str
    total_cost: float
    breakdown: Dict[str, float]  # e.g., {"input": 0.001, "output": 0.002, "cache_read": 0.0001}


def calculate_cost(
    tracker: SimpleTokenUsageTracker,
    pricing: Optional[PricingConfig]
) -> Optional[ModelCost]:
    """
    Calculate cost for a single model's token usage.

    Args:
        tracker: Token usage tracker for a single model
        pricing: Pricing configuration (optional)

    Returns:
        ModelCost if pricing is available and tokens were used, None otherwise
    """
    if pricing is None:
        return None

    # Calculate cost components
    breakdown = {}
    total_cost = 0.0

    # Input tokens
    if pricing.input_tokens_per_million is not None and tracker.input_tokens > 0:
        input_cost = (tracker.input_tokens / 1_000_000) * pricing.input_tokens_per_million
        breakdown['input'] = input_cost
        total_cost += input_cost

    # Output tokens (includes reasoning tokens)
    if pricing.output_tokens_per_million is not None and tracker.output_tokens > 0:
        output_cost = (tracker.output_tokens / 1_000_000) * pricing.output_tokens_per_million
        breakdown['output'] = output_cost
        total_cost += output_cost

    # Cache read tokens
    if pricing.cache_read_tokens_per_million is not None and tracker.cache_read_tokens > 0:
        cache_cost = (tracker.cache_read_tokens / 1_000_000) * pricing.cache_read_tokens_per_million
        breakdown['cache_read'] = cache_cost
        total_cost += cache_cost

    # Only return cost if we calculated something
    if total_cost == 0.0:
        return None

    return ModelCost(
        currency=pricing.currency,
        total_cost=total_cost,
        breakdown=breakdown
    )


def format_cost(cost: Optional[ModelCost]) -> str:
    """
    Format cost for display.

    Examples:
        "$0.0123 USD"
        "$0.001234 USD" (if very small)
        "€0.0123 EUR" (for other currencies)

    Args:
        cost: ModelCost to format (optional)

    Returns:
        Formatted cost string, or empty string if cost is None
    """
    if cost is None:
        return ""

    # Determine precision based on magnitude
    if cost.total_cost >= 0.01:
        precision = 4  # e.g., "$0.0123"
    elif cost.total_cost >= 0.001:
        precision = 5  # e.g., "$0.00123"
    else:
        precision = 6  # e.g., "$0.000123"

    # Format with currency symbol (simple mapping for common currencies)
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    symbol = currency_symbols.get(cost.currency, cost.currency + " ")

    return f"{symbol}{cost.total_cost:.{precision}f} {cost.currency}"
