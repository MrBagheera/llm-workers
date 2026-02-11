import unittest

from llm_workers.config import EvalDefinition, CallDefinition
from llm_workers.expressions import EvaluationContext, JsonExpression
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers_evaluation.config import EvaluationTestConfig
from llm_workers_evaluation.evaluation import (
    EvaluationTest, EvaluationResults, TestResult,
    ConfidenceInterval, calculate_confidence_interval, get_t_critical
)
from tests.mocks import StubWorkersContext


class TestEvaluationTest(unittest.TestCase):
    """Test EvaluationTest class functionality."""

    def test_evaluation_scores_and_logs(self):
        """Test that evaluation scores and logs are correctly tracked across iterations."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        # Test A: Returns 0.2 * TEST_ITERATION and logs "iter_{TEST_ITERATION}"
        # Scores: 0.0, 0.2, 0.4 for 3 iterations -> average = 0.2
        test_config_a = EvaluationTestConfig(
            do=[
                CallDefinition(
                    call='log',
                    params=JsonExpression({"entry": "iter_${TEST_ITERATION}"})
                ),
                EvalDefinition(eval=JsonExpression("${TEST_ITERATION * 0.2}"))
            ]
        )

        # Test B: Returns 0.3 * TEST_ITERATION and logs "iter_{TEST_ITERATION}"
        # Scores: 0.0, 0.3, 0.6 for 3 iterations -> average = 0.3
        test_config_b = EvaluationTestConfig(
            do=[
                CallDefinition(
                    call='log',
                    params=JsonExpression({"entry": "iter_${TEST_ITERATION}"})
                ),
                EvalDefinition(eval=JsonExpression("${TEST_ITERATION * 0.3}"))
            ]
        )

        test_a = EvaluationTest(
            test_name='test_a',
            test_config=test_config_a,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        test_b = EvaluationTest(
            test_name='test_b',
            test_config=test_config_b,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        # Run both tests with 3 iterations
        result_a = test_a.run(token_tracker, iterations=3)
        result_b = test_b.run(token_tracker, iterations=3)

        # Validate Test A scores
        self.assertEqual(result_a.scores, {0: 0.0, 1: 0.2, 2: 0.4})
        self.assertAlmostEqual(result_a.mean_score, 0.2, places=6)

        # Validate Test B scores
        self.assertEqual(result_b.scores, {0: 0.0, 1: 0.3, 2: 0.6})
        self.assertAlmostEqual(result_b.mean_score, 0.3, places=6)

        # Validate logs for Test A
        self.assertEqual(result_a.logs, {
            0: ["iter_0"],
            1: ["iter_1"],
            2: ["iter_2"]
        })

        # Validate logs for Test B
        self.assertEqual(result_b.logs, {
            0: ["iter_0"],
            1: ["iter_1"],
            2: ["iter_2"]
        })

        # Validate final score calculation
        results = EvaluationResults()
        results.tests['test_a'] = result_a
        results.tests['test_b'] = result_b

        # Calculate final score: (0.2 + 0.3) / 2 = 0.25
        tests_scores = [test.mean_score for test in results.tests.values()]
        mean_score = sum(tests_scores) / len(tests_scores)
        self.assertAlmostEqual(mean_score, 0.25, places=6)

    def test_evaluation_multiple_logs_per_iteration(self):
        """Test that multiple log entries per iteration are correctly captured."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        test_config = EvaluationTestConfig(
            do=[
                CallDefinition(
                    call='log',
                    params=JsonExpression({"entry": "first_${TEST_ITERATION}"})
                ),
                CallDefinition(
                    call='log',
                    params=JsonExpression({"entry": "second_${TEST_ITERATION}"})
                ),
                EvalDefinition(eval=JsonExpression(0.5))
            ]
        )

        test = EvaluationTest(
            test_name='multi_log_test',
            test_config=test_config,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        result = test.run(token_tracker, iterations=2)

        # Validate multiple logs per iteration
        self.assertEqual(result.logs, {
            0: ["first_0", "second_0"],
            1: ["first_1", "second_1"]
        })

        # Validate scores
        self.assertEqual(result.scores, {0: 0.5, 1: 0.5})
        self.assertAlmostEqual(result.mean_score, 0.5, places=6)

    def test_evaluation_score_clamping(self):
        """Test that scores are clamped to [0.0, 1.0] range."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        # Test that returns scores outside [0, 1] range
        test_config = EvaluationTestConfig(
            do=EvalDefinition(eval=JsonExpression("${TEST_ITERATION - 1}"))  # -1, 0, 1, 2 for iterations 0-3
        )

        test = EvaluationTest(
            test_name='clamp_test',
            test_config=test_config,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        result = test.run(token_tracker, iterations=4)

        # Scores should be clamped: -1->0.0, 0->0.0, 1->1.0, 2->1.0
        self.assertEqual(result.scores, {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0})
        self.assertAlmostEqual(result.mean_score, 0.5, places=6)

    def test_evaluation_with_data(self):
        """Test that test-level data is accessible in evaluation context."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        test_config = EvaluationTestConfig(
            data={
                "multiplier": JsonExpression(0.1)
            },
            do=EvalDefinition(eval=JsonExpression("${TEST_ITERATION * multiplier}"))
        )

        test = EvaluationTest(
            test_name='data_test',
            test_config=test_config,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        result = test.run(token_tracker, iterations=5)

        # 0*0.1=0.0, 1*0.1=0.1, 2*0.1=0.2, 3*0.1=0.3, 4*0.1=0.4
        expected_scores = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        self.assertEqual(len(result.scores), len(expected_scores))
        for i, expected in expected_scores.items():
            self.assertAlmostEqual(result.scores[i], expected, places=6)
        self.assertAlmostEqual(result.mean_score, 0.2, places=6)


class TestTestResult(unittest.TestCase):
    """Test TestResult model."""

    def test_default_values(self):
        """Test TestResult default values."""
        result = TestResult()
        self.assertEqual(result.scores, {})
        self.assertIsNone(result.errors)
        self.assertIsNone(result.logs)
        self.assertEqual(result.mean_score, 0.0)
        self.assertIsNone(result.CI_95)


class TestEvaluationResults(unittest.TestCase):
    """Test EvaluationResults model."""

    def test_default_values(self):
        """Test EvaluationResults default values."""
        results = EvaluationResults()
        self.assertEqual(results.tests, {})
        self.assertEqual(results.mean_score, 0.0)
        self.assertIsNone(results.CI_95)


class TestConfidenceInterval(unittest.TestCase):
    """Test confidence interval calculations."""

    def test_ci_none_for_single_score(self):
        """Test that CI is None when fewer than 2 scores."""
        ci = calculate_confidence_interval([0.5])
        self.assertIsNone(ci)

    def test_ci_none_for_empty_list(self):
        """Test that CI is None for empty list."""
        ci = calculate_confidence_interval([])
        self.assertIsNone(ci)

    def test_ci_calculated_for_two_scores(self):
        """Test CI is calculated for 2 scores."""
        ci = calculate_confidence_interval([0.4, 0.6])
        self.assertIsNotNone(ci)
        mean = 0.5
        self.assertLess(ci.min, mean)
        self.assertGreater(ci.max, mean)

    def test_ci_bounds_reasonable(self):
        """Test CI bounds are reasonable (min < mean < max)."""
        scores = [0.3, 0.5, 0.7, 0.4, 0.6]
        mean = sum(scores) / len(scores)
        ci = calculate_confidence_interval(scores)
        self.assertIsNotNone(ci)
        self.assertLess(ci.min, mean)
        self.assertGreater(ci.max, mean)

    def test_ci_identical_scores(self):
        """Test CI with all identical scores gives min = max = mean."""
        scores = [0.5, 0.5, 0.5, 0.5]
        mean = 0.5
        ci = calculate_confidence_interval(scores)
        self.assertIsNotNone(ci)
        self.assertAlmostEqual(ci.min, mean, places=6)
        self.assertAlmostEqual(ci.max, mean, places=6)

    def test_ci_with_many_scores(self):
        """Test CI calculation with more than 30 scores (uses approximation)."""
        scores = [0.5 + 0.01 * i for i in range(50)]
        mean = sum(scores) / len(scores)
        ci = calculate_confidence_interval(scores)
        self.assertIsNotNone(ci)
        self.assertLess(ci.min, mean)
        self.assertGreater(ci.max, mean)

    def test_get_t_critical_table_lookup(self):
        """Test t-critical values from table."""
        # df=1 (n=2)
        self.assertAlmostEqual(get_t_critical(2), 12.706, places=3)
        # df=10 (n=11)
        self.assertAlmostEqual(get_t_critical(11), 2.228, places=3)
        # df=30 (n=31)
        self.assertAlmostEqual(get_t_critical(31), 2.042, places=3)

    def test_get_t_critical_approximation(self):
        """Test t-critical approximation for large samples."""
        # df=100 (n=101), should be close to 1.984
        t_crit = get_t_critical(101)
        self.assertGreater(t_crit, 1.96)
        self.assertLess(t_crit, 2.0)

    def test_get_t_critical_error_for_n_less_than_2(self):
        """Test that get_t_critical raises error for n < 2."""
        with self.assertRaises(ValueError):
            get_t_critical(1)
        with self.assertRaises(ValueError):
            get_t_critical(0)


class TestEvaluationTestCI(unittest.TestCase):
    """Test confidence interval in EvaluationTest results."""

    def test_ci_none_for_single_iteration(self):
        """Test that CI is None when running only 1 iteration."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        test_config = EvaluationTestConfig(
            do=EvalDefinition(eval=JsonExpression(0.5))
        )

        test = EvaluationTest(
            test_name='single_iter_test',
            test_config=test_config,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        result = test.run(token_tracker, iterations=1)

        self.assertEqual(result.mean_score, 0.5)
        self.assertIsNone(result.CI_95)

    def test_ci_calculated_for_multiple_iterations(self):
        """Test that CI is calculated for multiple iterations."""
        context = StubWorkersContext()
        token_tracker = CompositeTokenUsageTracker()
        parent_context = EvaluationContext()

        # Scores: 0.0, 0.2, 0.4, 0.6 for 4 iterations
        test_config = EvaluationTestConfig(
            do=EvalDefinition(eval=JsonExpression("${TEST_ITERATION * 0.2}"))
        )

        test = EvaluationTest(
            test_name='multi_iter_test',
            test_config=test_config,
            suite_evaluation_context=parent_context,
            parent_tools=[],
            context=context
        )

        result = test.run(token_tracker, iterations=4)

        # Mean = (0 + 0.2 + 0.4 + 0.6) / 4 = 0.3
        self.assertAlmostEqual(result.mean_score, 0.3, places=6)
        self.assertIsNotNone(result.CI_95)
        self.assertLess(result.CI_95.min, result.mean_score)
        self.assertGreater(result.CI_95.max, result.mean_score)


if __name__ == '__main__':
    unittest.main()
