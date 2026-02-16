from typing import Any, Optional, Generator

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from llm_workers.api import ExtendedExecutionTool, WorkerNotification
from llm_workers.config import ToolDefinition
from llm_workers.starlark import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.worker_utils import tool_with_definition

TEST_LOGS_KEY = '__LOGS'
TEST_ITERATION_KEY = 'TEST_ITERATION'


class FBetaScoreInput(BaseModel):
    true_positives: int = Field(..., description="Count of correctly identified errors")
    false_negatives: int = Field(..., description="Count of missed expected errors")
    false_positives: int = Field(..., description="Count of incorrectly reported errors")
    beta: float = Field(1.0, description="Beta value (default 1.0 for F1 score). >1 favors recall, <1 favors precision.")

def calculate_f_beta(true_positives: int, false_negatives: int, false_positives: int, beta: float = 1.0) -> float:
    """Calculates the F-beta score."""
    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    beta_sq = beta ** 2
    if (beta_sq * precision) + recall == 0:
        return 0.0

    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

f_beta_score_tool = tool_with_definition(
    tool = StructuredTool.from_function(
        func=calculate_f_beta,
        name="f_beta_score",
        description="Calculates F-beta score based on confusion matrix metrics.",
        args_schema=FBetaScoreInput
    ),
    tool_def=ToolDefinition(ui_hint=False)
)

class ScoreInput(BaseModel):
    true_positives: int = Field(..., description="Count of correctly identified errors")
    false_negatives: int = Field(..., description="Count of missed errors (High penalty)")
    false_positives: int = Field(..., description="Count of incorrect errors found (Low penalty)")
    w_fn: float = Field(2.0, description="Weight for False Negatives")
    w_fp: float = Field(0.5, description="Weight for False Positives")

# 2. Define the Logic Function
def calculate_linear_score(
        true_positives: int,
        false_negatives: int,
        false_positives: int,
        w_fn: float = 2.0,
        w_fp: float = 0.5
) -> float:
    """
    Calculates a linear quality score (0-100) prioritizing the detection of errors.
    """
    expected_errors = true_positives + false_negatives
    max_score = 100.0

    if expected_errors == 0:
        # Edge case: No errors expected. Deduct for false positives.
        penalty = float(false_positives) * (w_fp * 10.0)
    else:
        weighted_penalty = (w_fn * false_negatives) + (w_fp * false_positives)
        penalty = (weighted_penalty / expected_errors) * 100.0

    return max(0.0, max_score - penalty)

linear_score_tool = tool_with_definition(
    tool = StructuredTool.from_function(
        func=calculate_linear_score,
        name="linear_score",
        description="Calculates a 0-100 quality score comparing found errors against expected errors.",
        args_schema=ScoreInput,
    ),
    tool_def=ToolDefinition(ui_hint=False)
)
