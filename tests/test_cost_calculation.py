"""Tests for cost calculation module."""

import unittest
from llm_workers.config import PricingConfig
from llm_workers.cost_calculation import ModelCost, calculate_cost, format_cost
from llm_workers.token_tracking import SimpleTokenUsageTracker


class TestCalculateCost(unittest.TestCase):
    """Tests for calculate_cost function."""

    def test_calculate_cost_with_complete_pricing(self):
        """Test cost calculation with all pricing fields configured."""
        # Create tracker with various token counts
        tracker = SimpleTokenUsageTracker()
        tracker.input_tokens = 1000
        tracker.output_tokens = 2000
        tracker.cache_read_tokens = 500

        # Create pricing config
        pricing = PricingConfig(
            currency="USD",
            input_tokens_per_million=1.0,
            output_tokens_per_million=3.0,
            cache_read_tokens_per_million=0.1
        )

        # Calculate cost
        cost = calculate_cost(tracker, pricing)

        # Verify result
        self.assertIsNotNone(cost)
        self.assertEqual(cost.currency, "USD")
        # Expected: (1000/1M * 1.0) + (2000/1M * 3.0) + (500/1M * 0.1)
        #         = 0.001 + 0.006 + 0.00005 = 0.00705
        self.assertAlmostEqual(cost.total_cost, 0.00705, places=6)
        self.assertIn('input', cost.breakdown)
        self.assertIn('output', cost.breakdown)
        self.assertIn('cache_read', cost.breakdown)

    def test_calculate_cost_with_partial_pricing(self):
        """Test cost calculation with only some pricing fields configured."""
        # Create tracker
        tracker = SimpleTokenUsageTracker()
        tracker.input_tokens = 1000
        tracker.output_tokens = 2000
        tracker.cache_read_tokens = 500

        # Create pricing config with only input/output prices
        pricing = PricingConfig(
            currency="USD",
            input_tokens_per_million=1.0,
            output_tokens_per_million=3.0,
            cache_read_tokens_per_million=None  # Not priced
        )

        # Calculate cost
        cost = calculate_cost(tracker, pricing)

        # Verify result - cache_read should be ignored
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost.total_cost, 0.007, places=6)
        self.assertIn('input', cost.breakdown)
        self.assertIn('output', cost.breakdown)
        self.assertNotIn('cache_read', cost.breakdown)

    def test_calculate_cost_no_pricing(self):
        """Test cost calculation with no pricing configured."""
        # Create tracker with tokens
        tracker = SimpleTokenUsageTracker()
        tracker.input_tokens = 1000
        tracker.output_tokens = 2000

        # Calculate cost with None pricing
        cost = calculate_cost(tracker, None)

        # Verify result is None
        self.assertIsNone(cost)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        # Create empty tracker
        tracker = SimpleTokenUsageTracker()

        # Create pricing config
        pricing = PricingConfig(
            currency="USD",
            input_tokens_per_million=1.0,
            output_tokens_per_million=3.0
        )

        # Calculate cost
        cost = calculate_cost(tracker, pricing)

        # Verify result is None (no tokens = no cost)
        self.assertIsNone(cost)

    def test_calculate_cost_only_input_tokens(self):
        """Test cost calculation with only input tokens."""
        # Create tracker with only input tokens
        tracker = SimpleTokenUsageTracker()
        tracker.input_tokens = 1000

        # Create pricing config
        pricing = PricingConfig(
            currency="EUR",
            input_tokens_per_million=2.0,
            output_tokens_per_million=6.0
        )

        # Calculate cost
        cost = calculate_cost(tracker, pricing)

        # Verify result
        self.assertIsNotNone(cost)
        self.assertEqual(cost.currency, "EUR")
        self.assertAlmostEqual(cost.total_cost, 0.002, places=6)
        self.assertIn('input', cost.breakdown)
        self.assertNotIn('output', cost.breakdown)


class TestFormatCost(unittest.TestCase):
    """Tests for format_cost function."""

    def test_format_cost_medium_magnitude(self):
        """Test formatting cost with medium magnitude (>= 0.01)."""
        cost = ModelCost(
            currency="USD",
            total_cost=0.0123,
            breakdown={}
        )

        formatted = format_cost(cost)

        # Should use 4 decimal places for costs >= 0.01
        self.assertEqual(formatted, "$0.0123 USD")

    def test_format_cost_small_magnitude(self):
        """Test formatting cost with small magnitude (>= 0.001)."""
        cost = ModelCost(
            currency="USD",
            total_cost=0.00456,
            breakdown={}
        )

        formatted = format_cost(cost)

        # Should use 5 decimal places for costs >= 0.001
        self.assertEqual(formatted, "$0.00456 USD")

    def test_format_cost_very_small_magnitude(self):
        """Test formatting cost with very small magnitude (< 0.001)."""
        cost = ModelCost(
            currency="USD",
            total_cost=0.000123,
            breakdown={}
        )

        formatted = format_cost(cost)

        # Should use 6 decimal places for costs < 0.001
        self.assertEqual(formatted, "$0.000123 USD")

    def test_format_cost_eur(self):
        """Test formatting cost with EUR currency."""
        cost = ModelCost(
            currency="EUR",
            total_cost=0.0234,
            breakdown={}
        )

        formatted = format_cost(cost)

        self.assertEqual(formatted, "€0.0234 EUR")

    def test_format_cost_gbp(self):
        """Test formatting cost with GBP currency."""
        cost = ModelCost(
            currency="GBP",
            total_cost=0.0345,
            breakdown={}
        )

        formatted = format_cost(cost)

        self.assertEqual(formatted, "£0.0345 GBP")

    def test_format_cost_jpy(self):
        """Test formatting cost with JPY currency."""
        cost = ModelCost(
            currency="JPY",
            total_cost=1.2345,
            breakdown={}
        )

        formatted = format_cost(cost)

        self.assertEqual(formatted, "¥1.2345 JPY")

    def test_format_cost_unknown_currency(self):
        """Test formatting cost with unknown currency."""
        cost = ModelCost(
            currency="XYZ",
            total_cost=0.0123,
            breakdown={}
        )

        formatted = format_cost(cost)

        # Should use currency code as prefix with space
        self.assertEqual(formatted, "XYZ 0.0123 XYZ")

    def test_format_cost_none(self):
        """Test formatting None cost."""
        formatted = format_cost(None)

        # Should return empty string
        self.assertEqual(formatted, "")


if __name__ == '__main__':
    unittest.main()
