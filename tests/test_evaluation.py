import unittest

from llm_workers.config import EvalDefinition, CallDefinition
from llm_workers.expressions import EvaluationContext, JsonExpression
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers_evaluation.config import EvaluationTestConfig
from llm_workers_evaluation.evaluation import EvaluationTest, EvaluationResults, TestResult
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
        self.assertAlmostEqual(result_a.average_score, 0.2, places=6)

        # Validate Test B scores
        self.assertEqual(result_b.scores, {0: 0.0, 1: 0.3, 2: 0.6})
        self.assertAlmostEqual(result_b.average_score, 0.3, places=6)

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
        tests_scores = [test.average_score for test in results.tests.values()]
        final_score = sum(tests_scores) / len(tests_scores)
        self.assertAlmostEqual(final_score, 0.25, places=6)

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
        self.assertAlmostEqual(result.average_score, 0.5, places=6)

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
        self.assertAlmostEqual(result.average_score, 0.5, places=6)

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
        self.assertAlmostEqual(result.average_score, 0.2, places=6)


class TestTestResult(unittest.TestCase):
    """Test TestResult model."""

    def test_default_values(self):
        """Test TestResult default values."""
        result = TestResult()
        self.assertEqual(result.scores, {})
        self.assertIsNone(result.errors)
        self.assertIsNone(result.logs)
        self.assertEqual(result.average_score, 0.0)


class TestEvaluationResults(unittest.TestCase):
    """Test EvaluationResults model."""

    def test_default_values(self):
        """Test EvaluationResults default values."""
        results = EvaluationResults()
        self.assertEqual(results.tests, {})
        self.assertEqual(results.final_score, 0.0)


if __name__ == '__main__':
    unittest.main()
