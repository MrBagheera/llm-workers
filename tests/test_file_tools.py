import os
import tempfile
import unittest

from langchain_core.tools.base import ToolException

from llm_workers.tools.fs import (
    ReadFileTool,
    EditFileTool,
    GlobFilesTool,
    GrepFilesTool,
    FileInfoTool,
)


class TestReadFileTool(unittest.TestCase):

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        # Create a 10-line file for testing
        with open(self.test_file, 'w') as f:
            for i in range(1, 11):
                f.write(f"Line {i}\n")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_read_entire_file(self):
        """Test reading entire file."""
        tool = ReadFileTool()
        result = tool._run(self.test_file, lines=1000)

        self.assertIn("Line 1", result)
        self.assertIn("Line 10", result)

    def test_read_with_offset_from_start(self):
        """Test reading with positive offset (from start)."""
        tool = ReadFileTool()
        result = tool._run(self.test_file, offset=5, lines=3)

        # Should read lines 6, 7, 8 (0-indexed offset 5 means starting at line 6)
        self.assertIn("Line 6", result)
        self.assertIn("Line 7", result)
        self.assertIn("Line 8", result)
        self.assertNotIn("Line 5", result)
        self.assertNotIn("Line 9", result)

    def test_read_with_negative_offset(self):
        """Test reading with negative offset (from end)."""
        tool = ReadFileTool()
        result = tool._run(self.test_file, offset=-3, lines=2)

        # Should read lines 8, 9 (last 3 lines are 8, 9, 10)
        self.assertIn("Line 8", result)
        self.assertIn("Line 9", result)
        self.assertNotIn("Line 7", result)
        self.assertNotIn("Line 10", result)

    def test_read_with_line_numbers(self):
        """Test reading with line numbers enabled."""
        tool = ReadFileTool()
        result = tool._run(self.test_file, offset=0, lines=3, show_line_numbers=True)

        self.assertIn("1:", result)
        self.assertIn("2:", result)
        self.assertIn("3:", result)

    def test_read_with_offset_and_line_numbers(self):
        """Test that line numbers are correct with offset."""
        tool = ReadFileTool()
        result = tool._run(self.test_file, offset=4, lines=2, show_line_numbers=True)

        # Offset 4 means starting at line 5 (1-indexed)
        self.assertIn("5:", result)
        self.assertIn("6:", result)
        self.assertNotIn("4:", result)

    def test_needs_confirmation(self):
        """Test confirmation requirements."""
        tool = ReadFileTool()

        # Relative path should not need confirmation
        self.assertFalse(tool.needs_confirmation({"path": "test.txt"}))

        # Absolute path should need confirmation
        self.assertTrue(tool.needs_confirmation({"path": "/tmp/test.txt"}))

        # Path with .. should need confirmation
        self.assertTrue(tool.needs_confirmation({"path": "../test.txt"}))

    def test_file_not_found(self):
        """Test error handling for nonexistent file."""
        tool = ReadFileTool()

        with self.assertRaises(ToolException) as context:
            tool._run(os.path.join(self.temp_dir, "nonexistent.txt"), lines=100)

        self.assertIn("Error reading file", str(context.exception))


class TestEditFileTool(unittest.TestCase):

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("Hello World\nThis is a test\nHello again")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_replacement(self):
        """Test basic string replacement."""
        tool = EditFileTool()
        result = tool._run(self.test_file, "World", "Universe")

        self.assertEqual(result["replacements"], 1)

        with open(self.test_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "Hello Universe\nThis is a test\nHello again")

    def test_replace_all(self):
        """Test replacing all occurrences."""
        tool = EditFileTool()
        result = tool._run(self.test_file, "Hello", "Hi", replace_all=True)

        self.assertEqual(result["replacements"], 2)

        with open(self.test_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "Hi World\nThis is a test\nHi again")

    def test_multiple_matches_without_replace_all_fails(self):
        """Test that multiple matches without replace_all raises error."""
        tool = EditFileTool()

        with self.assertRaises(ToolException) as context:
            tool._run(self.test_file, "Hello", "Hi", replace_all=False)

        self.assertIn("Multiple matches", str(context.exception))

    def test_string_not_found(self):
        """Test error when string is not found."""
        tool = EditFileTool()

        with self.assertRaises(ToolException) as context:
            tool._run(self.test_file, "NotFound", "Replacement")

        self.assertIn("not found", str(context.exception))

    def test_expected_count_validation(self):
        """Test expected_count validation."""
        tool = EditFileTool()

        # Correct expected count
        result = tool._run(self.test_file, "Hello", "Hi", replace_all=True, expected_count=2)
        self.assertEqual(result["replacements"], 2)

        # Reset file
        with open(self.test_file, 'w') as f:
            f.write("Hello World\nThis is a test\nHello again")

        # Wrong expected count
        with self.assertRaises(ToolException) as context:
            tool._run(self.test_file, "Hello", "Hi", replace_all=True, expected_count=3)

        self.assertIn("Expected 3 replacements but found 2", str(context.exception))

    def test_needs_confirmation(self):
        """Test confirmation requirements."""
        tool = EditFileTool()

        # Relative path should not need confirmation
        self.assertFalse(tool.needs_confirmation({"path": "test.txt"}))

        # Absolute path should need confirmation
        self.assertTrue(tool.needs_confirmation({"path": "/tmp/test.txt"}))

        # Path with .. should need confirmation
        self.assertTrue(tool.needs_confirmation({"path": "../test.txt"}))


class TestGlobFilesTool(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        self.py_file1 = os.path.join(self.temp_dir, "test1.py")
        self.py_file2 = os.path.join(self.temp_dir, "test2.py")
        self.txt_file = os.path.join(self.temp_dir, "test.txt")
        self.hidden_file = os.path.join(self.temp_dir, ".hidden.py")

        # Create subdirectory with files
        self.sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.sub_dir)
        self.sub_py_file = os.path.join(self.sub_dir, "sub.py")

        for f in [self.py_file1, self.py_file2, self.txt_file,
                  self.hidden_file, self.sub_py_file]:
            with open(f, 'w') as file:
                file.write("test content")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_glob(self):
        """Test basic glob pattern."""
        tool = GlobFilesTool()
        result = tool._run("*.py", path=self.temp_dir)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(f.endswith('.py') for f in result))

    def test_recursive_glob(self):
        """Test recursive glob pattern."""
        tool = GlobFilesTool()
        result = tool._run("**/*.py", path=self.temp_dir)

        self.assertEqual(len(result), 3)  # 2 in root + 1 in subdir

    def test_hidden_files_excluded_by_default(self):
        """Test that hidden files are excluded by default."""
        tool = GlobFilesTool()
        result = tool._run("*.py", path=self.temp_dir)

        hidden_in_result = any('.hidden' in f for f in result)
        self.assertFalse(hidden_in_result)

    def test_include_hidden_files(self):
        """Test including hidden files."""
        tool = GlobFilesTool()
        # Note: glob("*.py") doesn't match dotfiles like .hidden.py by default
        # We need to explicitly match hidden files with a pattern
        result = tool._run(".*py", path=self.temp_dir, include_hidden=True)

        # Should find .hidden.py
        self.assertEqual(len(result), 1)
        self.assertTrue(any('.hidden' in f for f in result))

    def test_max_results(self):
        """Test max_results limit."""
        tool = GlobFilesTool()
        result = tool._run("**/*.py", path=self.temp_dir, max_results=2)

        self.assertEqual(len(result), 2)

    def test_needs_confirmation(self):
        """Test confirmation requirements."""
        tool = GlobFilesTool()

        # Relative path should not need confirmation
        self.assertFalse(tool.needs_confirmation({"path": ".", "pattern": "*.py"}))

        # Absolute path should need confirmation
        self.assertTrue(tool.needs_confirmation({"path": "/tmp", "pattern": "*.py"}))


class TestGrepFilesTool(unittest.TestCase):

    def setUp(self):
        """Create temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()

        self.file1 = os.path.join(self.temp_dir, "file1.py")
        self.file2 = os.path.join(self.temp_dir, "file2.py")
        self.file3 = os.path.join(self.temp_dir, "file3.txt")

        with open(self.file1, 'w') as f:
            f.write("def hello():\n    print('Hello World')\n\ndef goodbye():\n    print('Goodbye')")

        with open(self.file2, 'w') as f:
            f.write("class Hello:\n    pass")

        with open(self.file3, 'w') as f:
            f.write("This is a text file\nWith some content")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_search(self):
        """Test basic regex search with directory."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=self.temp_dir)

        # Case-sensitive: matches "Hello World" and "class Hello:" but not "def hello():"
        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["files_searched"], 3)

    def test_case_insensitive_search(self):
        """Test case insensitive search."""
        tool = GrepFilesTool()
        result = tool._run("hello", files_glob=self.temp_dir, case_insensitive=True)

        self.assertEqual(result["total_matches"], 3)

    def test_files_glob_single_file(self):
        """Test files_glob with a single file path."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=self.file1)

        # Only matches "Hello World" in file1.py, not "def hello():"
        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["files_searched"], 1)

    def test_files_glob_directory(self):
        """Test files_glob with a directory path (no glob characters)."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=self.temp_dir)

        # Should search all files in directory
        self.assertEqual(result["files_searched"], 3)
        self.assertEqual(result["total_matches"], 2)

    def test_files_glob_pattern(self):
        """Test files_glob with glob pattern."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=os.path.join(self.temp_dir, "*.py"))

        # Should only search .py files
        self.assertEqual(result["files_searched"], 2)
        self.assertEqual(result["total_matches"], 2)

    def test_context_lines_separate(self):
        """Test separate lines_before and lines_after parameters."""
        tool = GrepFilesTool()
        result = tool._run("print", files_glob=self.file1, lines_before=1, lines_after=1)

        match = result["matches"][0]
        self.assertIn("context_before", match)
        self.assertIn("context_after", match)

    def test_asymmetric_context(self):
        """Test different values for lines_before and lines_after."""
        tool = GrepFilesTool()
        result = tool._run("print", files_glob=self.file1, lines_before=2, lines_after=1)

        # First print is on line 2, so context_before should have 1 line (line 1)
        match = result["matches"][0]
        self.assertIn("context_before", match)
        self.assertIn("context_after", match)
        # lines_before=2 but line 2 only has 1 line before it
        self.assertLessEqual(len(match["context_before"]), 2)
        self.assertLessEqual(len(match["context_after"]), 1)

    def test_context_only_before(self):
        """Test that context_after is absent when only lines_before is specified."""
        tool = GrepFilesTool()
        result = tool._run("print", files_glob=self.file1, lines_before=1, lines_after=0)

        match = result["matches"][0]
        self.assertIn("context_before", match)
        self.assertNotIn("context_after", match)

    def test_context_only_after(self):
        """Test that context_before is absent when only lines_after is specified."""
        tool = GrepFilesTool()
        result = tool._run("print", files_glob=self.file1, lines_before=0, lines_after=1)

        match = result["matches"][0]
        self.assertNotIn("context_before", match)
        self.assertIn("context_after", match)

    def test_no_context(self):
        """Test that context fields are absent when no context is requested."""
        tool = GrepFilesTool()
        result = tool._run("print", files_glob=self.file1)

        match = result["matches"][0]
        self.assertNotIn("context_before", match)
        self.assertNotIn("context_after", match)

    def test_output_mode_files_only(self):
        """Test files_only output mode."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=self.temp_dir, output_mode="files_only")

        self.assertIn("files", result)
        self.assertNotIn("matches", result)
        self.assertEqual(len(result["files"]), 2)

    def test_output_mode_count(self):
        """Test count output mode."""
        tool = GrepFilesTool()
        result = tool._run("Hello", files_glob=self.temp_dir, output_mode="count")

        self.assertIn("files_with_matches", result)
        self.assertEqual(result["files_with_matches"], 2)

    def test_max_results(self):
        """Test max_results limit."""
        tool = GrepFilesTool()
        result = tool._run(".", files_glob=self.temp_dir, max_results=2)

        self.assertEqual(len(result["matches"]), 2)
        self.assertGreater(result["total_matches"], 2)

    def test_invalid_regex(self):
        """Test error handling for invalid regex."""
        tool = GrepFilesTool()

        with self.assertRaises(ToolException) as context:
            tool._run("[invalid", files_glob=self.temp_dir)

        self.assertIn("Invalid regex", str(context.exception))

    def test_needs_confirmation(self):
        """Test confirmation requirements."""
        tool = GrepFilesTool()

        self.assertFalse(tool.needs_confirmation({"files_glob": ".", "pattern": "test"}))
        self.assertTrue(tool.needs_confirmation({"files_glob": "/tmp", "pattern": "test"}))


class TestFileInfoTool(unittest.TestCase):

    def setUp(self):
        """Create temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("Test content")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_file_info(self):
        """Test getting file info."""
        tool = FileInfoTool()
        result = tool._run(self.test_file)

        self.assertTrue(result["exists"])
        self.assertEqual(result["type"], "file")
        self.assertIn("size", result)
        self.assertIn("permissions", result)
        self.assertIn("owner", result)
        self.assertIn("modified", result)
        self.assertIn("is_readable", result)
        self.assertIn("mime_type", result)

    def test_directory_info(self):
        """Test getting directory info."""
        tool = FileInfoTool()
        result = tool._run(self.temp_dir)

        self.assertTrue(result["exists"])
        self.assertEqual(result["type"], "directory")
        self.assertNotIn("mime_type", result)

    def test_nonexistent_file(self):
        """Test info for nonexistent file."""
        tool = FileInfoTool()
        result = tool._run(os.path.join(self.temp_dir, "nonexistent.txt"))

        self.assertFalse(result["exists"])
        self.assertNotIn("type", result)

    def test_permissions_format(self):
        """Test permissions string format."""
        tool = FileInfoTool()
        result = tool._run(self.test_file)

        # Should be 9 characters like "rw-r--r--"
        self.assertEqual(len(result["permissions"]), 9)
        for char in result["permissions"]:
            self.assertIn(char, "rwx-")

    def test_no_confirmation_needed(self):
        """Test that file info never needs confirmation (read-only)."""
        tool = FileInfoTool()

        # Even absolute paths shouldn't need confirmation
        self.assertFalse(tool.needs_confirmation({"path": "/tmp/test.txt"}))
        self.assertFalse(tool.needs_confirmation({"path": "test.txt"}))


if __name__ == '__main__':
    unittest.main()
