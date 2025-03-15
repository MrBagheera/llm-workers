import unittest

import yaml

from llm_workers.config import CallDefinition, ResultDefinition, MatchDefinition, ToolDefinition


class TestCustomToolDefinition(unittest.TestCase):

    def test_single_step_body(self):
        definition = ToolDefinition(**yaml.safe_load("""
name: ExampleTool
description: This is an example tool
input:
  - name: input1
    description: This is the first input
    type: string
  - name: input2
    description: This is the second input
    type: integer
    default: 42
body:
    call: some_function
    params:
      key1: value1
      key2: 42
    model: some_model
return_direct: false"""))
        assert definition.name == "ExampleTool"
        assert definition.description == "This is an example tool"
        assert len(definition.input) == 2
        assert definition.return_direct is False
        assert isinstance(definition.body, CallDefinition)
        assert definition.body.call == "some_function"
        assert definition.body.params == {"key1": "value1", "key2": 42}

    def test_multi_step_body(self):
        definition = ToolDefinition(**yaml.safe_load("""
name: ExampleTool
description: This is an example tool
input:
  - name: input1
    description: This is the first input
    type: string
  - name: input2
    description: This is the second input
    type: integer
    default: 42
body:
  - call: some_function
    params:
      key1: value1
      key2: 42
    model: some_model
  - result: value2      
return_direct: true"""))
        assert definition.name == "ExampleTool"
        assert definition.description == "This is an example tool"
        assert len(definition.input) == 2
        assert definition.return_direct is True
        assert isinstance(definition.body, list)
        assert len(definition.body) == 2
        assert isinstance(definition.body[0], CallDefinition)
        assert isinstance(definition.body[1], ResultDefinition)

    def test_match_body(self):
        definition = ToolDefinition(**yaml.safe_load("""
name: ExampleTool
description: This is an example tool
input:
  - name: input1
    description: This is the first input
    type: string
body:
  match: "{input1}"
  matchers:
    - case: Meaning of life
      then:
        result: 42
    - pattern: "[0-9]+"
      then:
        result: "a number"
    - pattern: "https?://([^/]+)/?.*"
      then:
        result: "an URL pointing to {match0}"
  default:
    result: -1
"""))
        assert definition.name == "ExampleTool"
        assert definition.description == "This is an example tool"
        assert len(definition.input) == 1
        assert isinstance(definition.body, MatchDefinition)
        assert definition.body.match == "{input1}"
        assert len(definition.body.matchers) == 3
        assert definition.body.default.result == -1
        assert definition.body.matchers[0].case == "Meaning of life"
        assert definition.body.matchers[0].then.result == 42
        assert definition.body.matchers[1].pattern == "[0-9]+"
        assert definition.body.matchers[1].then.result == "a number"
        assert definition.body.matchers[2].pattern == "https?://([^/]+)/?.*"
        assert definition.body.matchers[2].then.result == "an URL pointing to {match0}"