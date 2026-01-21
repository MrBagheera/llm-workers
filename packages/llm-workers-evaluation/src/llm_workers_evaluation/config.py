"""Pydantic models for evaluation suite YAML configuration."""

from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from llm_workers.config import BodyDefinition, ToolsDefinitionStatement
from llm_workers.expressions import JsonExpression


class EvaluationTestConfig(BaseModel):
    """Configuration for a single evaluation test."""
    model_config = ConfigDict(extra='forbid')
    data: Dict[str, JsonExpression] = {}
    tools: List[ToolsDefinitionStatement] = []
    do: BodyDefinition  # Must return float score [0.0, 1.0]


class EvaluationSharedConfig(BaseModel):
    """Shared configuration applied to all suites and tests."""
    model_config = ConfigDict(extra='forbid')
    data: Dict[str, JsonExpression] = {}
    tools: List[ToolsDefinitionStatement] = []


class EvaluationSuiteFile(BaseModel):
    """Root configuration for an evaluation suite file."""
    model_config = ConfigDict(extra='forbid')
    shared: EvaluationSharedConfig = EvaluationSharedConfig()
    iterations: int
    tests: Dict[str, EvaluationTestConfig]
