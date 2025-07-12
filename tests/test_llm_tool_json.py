import unittest
from llm_workers.tools.llm_tool import extract_json_blocks


class TestLLMToolJSONExtraction(unittest.TestCase):

    def test_extract_json_none_returns_original(self):
        text = "Some text with JSON: {'key': 'value'}"
        result = extract_json_blocks(text, None)
        self.assertEqual(result, text)
        
        result = extract_json_blocks(text, "none")
        self.assertEqual(result, text)
        
        result = extract_json_blocks(text, False)
        self.assertEqual(result, text)

    def test_extract_json_first_from_code_blocks(self):
        text = """Here's some JSON:
```json
{"first": "block"}
```
And another:
```json
{"second": "block"}
```"""
        result = extract_json_blocks(text, "first")
        self.assertEqual(result, '{"first": "block"}')
        
        result = extract_json_blocks(text, True)
        self.assertEqual(result, '{"first": "block"}')

    def test_extract_json_last_from_code_blocks(self):
        text = """Here's some JSON:
```json
{"first": "block"}
```
And another:
```json
{"second": "block"}
```"""
        result = extract_json_blocks(text, "last")
        self.assertEqual(result, '{"second": "block"}')

    def test_extract_json_all_from_code_blocks(self):
        text = """Here's some JSON:
```json
{"first": "block"}
```
And another:
```json
{"second": "block"}
```"""
        result = extract_json_blocks(text, "all")
        self.assertEqual(result, '["{\\"first\\": \\"block\\"}", "{\\"second\\": \\"block\\"}"]')

    def test_extract_json_fallback_when_no_json(self):
        text = "This is just plain text with no JSON"
        result = extract_json_blocks(text, "first")
        self.assertEqual(result, text)
        
        result = extract_json_blocks(text, "last")
        self.assertEqual(result, text)
        
        result = extract_json_blocks(text, "all")
        self.assertEqual(result, text)

    def test_extract_json_complex_nested(self):
        text = """Response with nested JSON:
```json
{
  "results": [
    {"bullet": "First point", "importance": 5, "references": ["ref1"]},
    {"bullet": "Second point", "importance": 3, "references": ["ref2", "ref3"]}
  ]
}
```"""
        result = extract_json_blocks(text, "first")
        expected = """{
  "results": [
    {"bullet": "First point", "importance": 5, "references": ["ref1"]},
    {"bullet": "Second point", "importance": 3, "references": ["ref2", "ref3"]}
  ]
}"""
        self.assertEqual(result, expected)

    def test_extract_json_mixed_code_blocks(self):
        text = """Some code:
```
{"not_json_block": "value"}
```
And actual JSON:
```json
{"json_block": "value"}
```"""
        result = extract_json_blocks(text, "first")
        self.assertEqual(result, '{"json_block": "value"}')

    def test_extract_json_array_first(self):
        text = """Arrays work too:
```json
[
  {"bullet": "First point", "importance": 5},
  {"bullet": "Second point", "importance": 3}
]
```"""
        result = extract_json_blocks(text, True)
        expected = """[
  {"bullet": "First point", "importance": 5},
  {"bullet": "Second point", "importance": 3}
]"""
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()