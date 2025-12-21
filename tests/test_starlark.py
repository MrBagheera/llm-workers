import unittest

from llm_workers.starlark import StarlarkEval, StarlarkExec


class TestStarlarkEval(unittest.TestCase):
    """Tests for StarlarkEval - expression evaluation."""

    def test_simple_arithmetic(self):
        """Test simple arithmetic expressions."""
        evaluator = StarlarkEval("1 + 2 * 3")
        result = evaluator.run({}, {})
        self.assertEqual(result, 7)

    def test_string_operations(self):
        """Test string operations."""
        evaluator = StarlarkEval("'hello' + ' ' + 'world'")
        result = evaluator.run({}, {})
        self.assertEqual(result, "hello world")

    def test_list_operations(self):
        """Test list operations."""
        evaluator = StarlarkEval("[1, 2, 3] + [4, 5]")
        result = evaluator.run({}, {})
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_dict_operations(self):
        """Test dictionary operations."""
        evaluator = StarlarkEval("{'a': 1, 'b': 2}")
        result = evaluator.run({}, {})
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_access_global_vars(self):
        """Test accessing variables from global_vars."""
        evaluator = StarlarkEval("x + y")
        result = evaluator.run({'x': 10, 'y': 20}, {})
        self.assertEqual(result, 30)

    def test_call_allowed_function(self):
        """Test calling an allowed function."""
        def add(a, b):
            return a + b

        evaluator = StarlarkEval("add(5, 3)")
        result = evaluator.run({}, {'add': add})
        self.assertEqual(result, 8)

    def test_struct_creation(self):
        """Test creating and using struct in expressions."""
        evaluator = StarlarkEval("struct(x=1, y=2).x")
        result = evaluator.run({}, {})
        self.assertEqual(result, 1)

    def test_reject_lambda(self):
        """Test that lambda expressions are rejected."""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkEval("lambda x: x + 1")
        self.assertIn("lambda", str(cm.exception).lower())

    def test_reject_while_loop(self):
        """Test that while loops are rejected in eval mode."""
        # While loops require exec mode, but let's ensure eval syntax is checked
        with self.assertRaises(SyntaxError):
            StarlarkEval("x = 0\nwhile x < 5: x += 1")

    def test_reject_class_definition(self):
        """Test that class definitions are rejected."""
        with self.assertRaises(SyntaxError):
            StarlarkEval("class Foo: pass")

    def test_reject_import(self):
        """Test that import statements are rejected."""
        with self.assertRaises(SyntaxError):
            StarlarkEval("import os")

    def test_reject_import_from(self):
        """Test that from...import statements are rejected."""
        with self.assertRaises(SyntaxError):
            StarlarkEval("from os import path")

    def test_empty_allowed_functions(self):
        """Test that empty allowed_functions list works correctly."""
        evaluator = StarlarkEval("1 + 1")
        result = evaluator.run({}, {})
        self.assertEqual(result, 2)

    def test_sanitized_data_access(self):
        """Test that objects are sanitized and converted to structs."""
        class CustomObject:
            def __init__(self):
                self.public_value = 42
                self._private_value = 99

        obj = CustomObject()
        evaluator = StarlarkEval("obj.public_value")
        result = evaluator.run({'obj': obj}, {})
        self.assertEqual(result, 42)

    def test_block_private_attribute_access(self):
        """Test that private attribute access is blocked at compile time."""
        # RestrictedPython blocks private attributes at compile time
        with self.assertRaises(SyntaxError):
            StarlarkEval("obj._private")

    def test_complex_nested_expression(self):
        """Test complex nested expressions."""
        evaluator = StarlarkEval("[x * 2 for x in range(5) if x % 2 == 0]")
        result = evaluator.run({}, {})
        self.assertEqual(result, [0, 4, 8])

    def test_dict_access(self):
        """Test dictionary access in expressions."""
        evaluator = StarlarkEval("data['key']")
        result = evaluator.run({'data': {'key': 'value'}}, {})
        self.assertEqual(result, 'value')

    def test_list_indexing(self):
        """Test list indexing in expressions."""
        evaluator = StarlarkEval("items[2]")
        result = evaluator.run({'items': [10, 20, 30, 40]}, {})
        self.assertEqual(result, 30)

    # --- Alternative expression tests for DEFAULT_FUNCTIONS replacements ---

    def test_get_dict_alternative(self):
        """Test dict.get() as alternative to get_with_default for dicts."""
        evaluator = StarlarkEval("my_dict.get('key', 'default_value')")

        # Key exists
        result = evaluator.run({'my_dict': {'key': 'value'}}, {})
        self.assertEqual(result, 'value')

        # Key doesn't exist
        result = evaluator.run({'my_dict': {'other': 'value'}}, {})
        self.assertEqual(result, 'default_value')

    def test_get_list_alternative(self):
        """Test ternary expression as alternative to get_with_default for lists."""
        evaluator = StarlarkEval(
            "my_list[idx] if (idx >= 0 and idx < len(my_list)) else 'default'"
        )

        # Valid index
        result = evaluator.run({'my_list': ['a', 'b', 'c'], 'idx': 1}, {})
        self.assertEqual(result, 'b')

        # Out of bounds (too high)
        result = evaluator.run({'my_list': ['a', 'b', 'c'], 'idx': 10}, {})
        self.assertEqual(result, 'default')

        # Negative index (should use default based on condition)
        result = evaluator.run({'my_list': ['a', 'b', 'c'], 'idx': -1}, {})
        self.assertEqual(result, 'default')

    def test_merge_lists_alternative(self):
        """Test + operator as alternative to merge for lists."""
        evaluator = StarlarkEval("list1 + list2")

        result = evaluator.run({'list1': [1, 2], 'list2': [3, 4]}, {})
        self.assertEqual(result, [1, 2, 3, 4])

    def test_merge_dicts_alternative_unpacking(self):
        """Test dict unpacking as alternative to merge for dicts."""
        evaluator = StarlarkEval("{**dict1, **dict2}")

        result = evaluator.run({
            'dict1': {'a': 1, 'b': 2},
            'dict2': {'b': 3, 'c': 4}
        }, {})
        # dict2 values should override dict1 for duplicate keys
        self.assertEqual(result, {'a': 1, 'b': 3, 'c': 4})

    def test_merge_strings_alternative(self):
        """Test + operator as alternative to merge for strings."""
        evaluator = StarlarkEval("str1 + str2")

        result = evaluator.run({'str1': 'hello', 'str2': 'world'}, {})
        self.assertEqual(result, 'helloworld')

    def test_flatten_alternative_comprehension(self):
        """Test list comprehension as alternative to flatten (all sublists)."""
        evaluator = StarlarkEval(
            "[item for sublist in list_of_lists for item in sublist]"
        )

        result = evaluator.run({
            'list_of_lists': [[1, 2], [3, 4], [5]]
        }, {})
        self.assertEqual(result, [1, 2, 3, 4, 5])


class TestStarlarkExec(unittest.TestCase):
    """Tests for StarlarkExec - script execution."""

    def test_script_with_result_variable(self):
        """Test script that defines a result variable."""
        script = """
x = 10
y = 20
result = x + y
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 30)

    def test_script_with_run_function(self):
        """Test script that defines a run() function."""
        script = """
def run():
    return 42
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 42)

    def test_error_when_no_result_or_run(self):
        """Test that error is raised when neither result nor run() exists."""
        script = "x = 10"
        executor = StarlarkExec(script)
        with self.assertRaises(RuntimeError) as cm:
            executor.run({}, {})
        self.assertIn("result", str(cm.exception).lower())
        self.assertIn("run", str(cm.exception).lower())

    def test_access_global_vars(self):
        """Test accessing variables from global_vars."""
        script = """
result = a + b + c
"""
        executor = StarlarkExec(script)
        result = executor.run({'a': 1, 'b': 2, 'c': 3}, {})
        self.assertEqual(result, 6)

    def test_call_allowed_functions(self):
        """Test calling allowed functions from script."""
        def multiply(x, y):
            return x * y

        script = """
result = multiply(6, 7)
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {'multiply': multiply})
        self.assertEqual(result, 42)

        script = """
result = secret_func()
"""
        executor = StarlarkExec(script)
        with self.assertRaises(Exception):
            executor.run({}, {'secret_func': secret_func})

    def test_if_statements(self):
        """Test if/elif/else statements."""
        script = """
def run():
    x = 10
    if x > 15:
        return "big"
    elif x > 5:
        return "medium"
    else:
        return "small"
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, "medium")

    def test_for_loops(self):
        """Test for loops."""
        script = """
def run():
    total = 0
    for i in range(5):
        total = total + i
    return total
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 10)

    def test_list_comprehensions(self):
        """Test list comprehensions."""
        script = """
result = [x * x for x in range(5)]
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, [0, 1, 4, 9, 16])

    def test_dict_comprehensions(self):
        """Test dict comprehensions."""
        script = """
result = {x: x * 2 for x in range(3)}
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, {0: 0, 1: 2, 2: 4})

    def test_multiple_functions(self):
        """Test multiple function definitions."""
        script = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def run():
    return add(2, 3) + multiply(4, 5)
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 25)

    def test_function_with_parameters(self):
        """Test functions with parameters."""
        script = """
def calculate(x, y, z):
    return x * y + z

result = calculate(3, 4, 5)
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 17)

    def test_struct_creation_and_usage(self):
        """Test creating and using structs in scripts."""
        script = """
def run():
    person = struct(name="Alice", age=30)
    return person.name
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, "Alice")

    def test_reject_while_loop(self):
        """Test that while loops are rejected."""
        script = """
x = 0
while x < 5:
    x = x + 1
result = x
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("while", str(cm.exception).lower())

    def test_reject_class_definition(self):
        """Test that class definitions are rejected."""
        script = """
class MyClass:
    pass
result = 1
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("class", str(cm.exception).lower())

    def test_reject_import(self):
        """Test that import statements are rejected."""
        script = """
import os
result = 1
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("import", str(cm.exception).lower())

    def test_reject_try_except(self):
        """Test that try/except blocks are rejected."""
        script = """
try:
    result = 1
except:
    result = 0
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("try", str(cm.exception).lower())

    def test_reject_generators(self):
        """Test that generators/yield are rejected."""
        script = """
def gen():
    yield 1
    yield 2
result = list(gen())
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("yield", str(cm.exception).lower())

    def test_reject_direct_recursion(self):
        """Test that direct recursion is rejected."""
        script = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("recursion", str(cm.exception).lower())

    def test_reject_nested_functions(self):
        """Test that nested functions are rejected."""
        script = """
def outer():
    def inner():
        return 42
    return inner()

result = outer()
"""
        with self.assertRaises(SyntaxError) as cm:
            StarlarkExec(script)
        self.assertIn("nested", str(cm.exception).lower())

    def test_empty_allowed_functions(self):
        """Test that empty allowed_functions list works correctly."""
        script = """
result = 1 + 1
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 2)

    def test_sanitized_data_access(self):
        """Test that objects are sanitized and accessible in scripts."""
        class CustomObject:
            def __init__(self):
                self.value = 100
                self._hidden = 200

        obj = CustomObject()
        script = """
result = data.value
"""
        executor = StarlarkExec(script)
        result = executor.run({'data': obj}, {})
        self.assertEqual(result, 100)

    def test_block_private_attribute_access(self):
        """Test that private attribute access is blocked at compile time."""
        # RestrictedPython blocks private attributes at compile time
        script = """
result = obj._secret
"""
        with self.assertRaises(SyntaxError):
            StarlarkExec(script)

    def test_complex_multi_function_script(self):
        """Test complex script with multiple functions and logic."""
        script = """
def fibonacci(n):
    a, b = 0, 1
    result = []
    for i in range(n):
        result.append(a)
        a, b = b, a + b
    return result

def sum_list(lst):
    total = 0
    for item in lst:
        total = total + item
    return total

def run():
    fib = fibonacci(7)
    return sum_list(fib)
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        # fibonacci(7) = [0, 1, 1, 2, 3, 5, 8], sum = 20
        self.assertEqual(result, 20)

    def test_result_priority_over_run(self):
        """Test that result variable takes priority over run() function."""
        script = """
def run():
    return 999

result = 42
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {})
        self.assertEqual(result, 42)

    def test_function_calling_allowed_function(self):
        """Test user-defined function calling an allowed external function."""
        def external(x):
            return x * 10

        script = """
def process(val):
    return external(val) + 5

result = process(3)
"""
        executor = StarlarkExec(script)
        result = executor.run({}, {'external': external})
        self.assertEqual(result, 35)


if __name__ == "__main__":
    unittest.main()
