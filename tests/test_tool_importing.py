import asyncio
import unittest
from typing import List, Optional, Dict, Any

from langchain_core.tools import BaseTool

from llm_workers.config import ToolsReference
from tests.mocks import MockMCPWorkersContext, create_mock_mcp_tool, create_context_from_yaml


# Helper Functions


async def initialize_context_async(context: MockMCPWorkersContext) -> MockMCPWorkersContext:
    """Initialize context asynchronously (calls __aenter__).

    Args:
        context: MockMCPWorkersContext instance to initialize

    Returns:
        The initialized context
    """
    return await context.__aenter__()


def get_tool_names(tools: List[BaseTool]) -> List[str]:
    """Extract sorted list of tool names from tools list.

    Args:
        tools: List of BaseTool instances

    Returns:
        Sorted list of tool names
    """
    return sorted([tool.name for tool in tools])


def get_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    """Find tool by name in tools list.

    Args:
        tools: List of BaseTool instances
        name: Name of the tool to find

    Returns:
        BaseTool instance if found, None otherwise
    """
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def assert_tool_has_property(test_case: unittest.TestCase, tool: BaseTool,
                              property_name: str, expected_value: Any):
    """Assert that tool metadata contains expected property.

    Args:
        test_case: unittest.TestCase instance for assertions
        tool: BaseTool instance to check
        property_name: Name of property to check in tool_definition
        expected_value: Expected value of the property
    """
    test_case.assertIsNotNone(tool.metadata, f"Tool {tool.name} has no metadata")
    tool_def = tool.metadata.get('tool_definition')
    test_case.assertIsNotNone(tool_def, f"Tool {tool.name} has no tool_definition in metadata")
    actual_value = getattr(tool_def, property_name, None)
    test_case.assertEqual(expected_value, actual_value,
                          f"Tool {tool.name} property {property_name}")


# Base Test Class

class BaseToolImportTest(unittest.TestCase):
    """Base class with common setup and assertions for tool import tests."""

    def setUp(self):
        """Common setup for all tool import tests."""
        self.maxDiff = None  # Show full diffs in assertions

    @staticmethod
    def create_and_initialize_context(yaml_str: str,
                                      mock_mcp_tools: Dict[str, List[BaseTool]] = None) -> MockMCPWorkersContext:
        """Create and initialize context from YAML synchronously.

        Args:
            yaml_str: YAML configuration string
            mock_mcp_tools: Optional dictionary mapping MCP server names to lists of mock tools

        Returns:
            Initialized MockMCPWorkersContext instance
        """
        context = create_context_from_yaml(yaml_str, mock_mcp_tools)
        return asyncio.run(initialize_context_async(context))

    def assert_tools_available(self, context: MockMCPWorkersContext,
                                scope: str, tool_refs: List[str],
                                expected_names: List[str]):
        """Assert that specific tools are available in scope.

        Args:
            context: MockMCPWorkersContext instance
            scope: Scope name ('chat', 'shared tools', etc.)
            tool_refs: List of tool references to retrieve
            expected_names: Expected sorted list of tool names
        """
        tools = context.get_tools(scope, tool_refs)
        actual_names = get_tool_names(tools)
        self.assertEqual(expected_names, actual_names)

    def assert_tool_metadata(self, tool: BaseTool, **expected_props):
        """Assert tool has expected metadata properties.

        Args:
            tool: BaseTool instance to check
            **expected_props: Property names and expected values
        """
        for prop_name, expected_value in expected_props.items():
            assert_tool_has_property(self, tool, prop_name, expected_value)


# Test Category 1: Import to Shared Tools and Reference from Chat

class TestImportToSharedToolsAndReference(BaseToolImportTest):
    """Test importing tools to shared tools section and referencing from chat."""

    def test_import_single_tool_to_shared_and_reference(self):
        """Import single tool to shared tools and reference it from chat."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    description: Read files
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is in shared tools
        self.assertIn('read_tool', context._tools)
        tool = context._tools['read_tool']
        self.assertEqual('read_tool', tool.name)
        self.assertEqual('Read files', tool.description)

        # Verify tool can be referenced from chat
        chat_tools = context.get_tools('chat', ['read_tool'])
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('read_tool', chat_tools[0].name)

    def test_import_single_tool_from_toolkit_to_shared_and_reference(self):
        """Import one tool from toolkit to shared tools and reference from chat."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.FilesystemToolkit/read_file
    name: my_read_tool
    description: Custom read tool from toolkit
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is in shared tools with custom name
        self.assertIn('my_read_tool', context._tools)
        tool = context._tools['my_read_tool']
        self.assertEqual('my_read_tool', tool.name)
        self.assertEqual('Custom read tool from toolkit', tool.description)

        # Verify tool can be referenced from chat
        chat_tools = context.get_tools('chat', ['my_read_tool'])
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('my_read_tool', chat_tools[0].name)

    def test_import_single_tool_from_mcp_to_shared_and_reference(self):
        """Import one tool from MCP server to shared tools and reference from chat."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'test_server': [
                create_mock_mcp_tool('mcp_read', 'Read from MCP'),
                create_mock_mcp_tool('mcp_write', 'Write to MCP'),
                create_mock_mcp_tool('mcp_list', 'List from MCP'),
            ]
        }

        yaml_config = """
mcp:
  test_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tool: mcp:test_server/mcp_read
    name: my_mcp_read
    description: Custom MCP read tool
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify tool is in shared tools with custom name
        self.assertIn('my_mcp_read', context._tools)
        tool = context._tools['my_mcp_read']
        self.assertEqual('my_mcp_read', tool.name)
        self.assertEqual('Custom MCP read tool', tool.description)

        # Verify tool can be referenced from chat
        chat_tools = context.get_tools('chat', ['my_mcp_read'])
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('my_mcp_read', chat_tools[0].name)

    def test_declare_custom_tool_in_shared_and_reference(self):
        """Declare custom tool in shared tools and reference from chat."""
        yaml_config = """
tools:
  - name: custom_tool
    description: A custom tool
    input:
      - name: query
        description: Search query
        type: str
    body:
      - result: "Query result for {query}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is in shared tools
        self.assertIn('custom_tool', context._tools)
        tool = context._tools['custom_tool']
        self.assertEqual('custom_tool', tool.name)
        self.assertEqual('A custom tool', tool.description)

        # Verify tool can be referenced from chat
        chat_tools = context.get_tools('chat', ['custom_tool'])
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('custom_tool', chat_tools[0].name)

    def test_import_toolkit_to_shared_and_reference(self):
        """Import entire toolkit to shared tools and reference from chat."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: 'fs_'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify toolkit tools are in shared tools with prefix
        shared_tool_names = list(context._tools.keys())
        self.assertTrue(any(name == 'fs_read_file' for name in shared_tool_names))
        self.assertTrue(any(name == 'fs_write_file' for name in shared_tool_names))

        # Verify tools can be referenced from chat with wildcard
        chat_tools = context.get_tools('chat', [ToolsReference(match=['fs_*'])])
        self.assertGreater(len(chat_tools), 0)
        # All tools should have fs_ prefix
        for tool in chat_tools:
            self.assertTrue(tool.name.startswith('fs_'))

    def test_import_mcp_server_to_shared_and_reference(self):
        """Import all MCP server tools to shared tools and reference from chat."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'test_server': [
                create_mock_mcp_tool('mcp_read', 'Read from MCP'),
                create_mock_mcp_tool('mcp_write', 'Write to MCP'),
                create_mock_mcp_tool('mcp_list', 'List from MCP'),
            ]
        }

        yaml_config = """
mcp:
  test_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tools: mcp:test_server
    prefix: 'server_'
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify MCP tools are in shared tools with prefix
        shared_tool_names = list(context._tools.keys())
        self.assertIn('server_mcp_read', shared_tool_names)
        self.assertIn('server_mcp_write', shared_tool_names)
        self.assertIn('server_mcp_list', shared_tool_names)

        # Verify tools can be referenced from chat with wildcard
        chat_tools = context.get_tools('chat', [ToolsReference(match=['server_*'])])
        self.assertEqual(3, len(chat_tools))
        # All tools should have server_ prefix
        for tool in chat_tools:
            self.assertTrue(tool.name.startswith('server_'))

    def test_auto_import_mcp_server_to_shared_and_reference(self):
        """MCP server with auto_import_scope: shared tools."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'auto_server': [
                create_mock_mcp_tool('auto_read', 'Auto-imported read'),
                create_mock_mcp_tool('auto_write', 'Auto-imported write'),
            ]
        }

        yaml_config = """
mcp:
  auto_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: shared tools
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify MCP tools are auto-imported to shared tools with server name prefix
        shared_tool_names = list(context._tools.keys())
        self.assertIn('auto_server_auto_read', shared_tool_names)
        self.assertIn('auto_server_auto_write', shared_tool_names)

        # Verify tools can be referenced from chat
        chat_tools = context.get_tools('chat', [ToolsReference(match=['auto_server_*'])])
        self.assertEqual(2, len(chat_tools))
        # All tools should have auto_server_ prefix
        for tool in chat_tools:
            self.assertTrue(tool.name.startswith('auto_server_'))


# Test Category 2: Import Directly to Chat Section

class TestImportDirectlyToChat(BaseToolImportTest):
    """Test importing tools directly in chat section."""

    def test_import_single_tool_directly_to_chat(self):
        """Import single tool directly in chat section."""
        yaml_config = """
chat:
  tools:
    - import_tool: llm_workers.tools.fs.ReadFileTool
      name: read_tool
      description: Read files directly
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is NOT in shared tools (imported directly to chat)
        self.assertNotIn('read_tool', context._tools)

        # Verify tool is accessible from chat by passing the chat config's tools
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('read_tool', chat_tools[0].name)
        self.assertEqual('Read files directly', chat_tools[0].description)

    def test_import_single_tool_from_toolkit_directly_to_chat(self):
        """Import one tool from toolkit directly in chat section."""
        yaml_config = """
chat:
  tools:
    - import_tool: llm_workers.tools.fs.FilesystemToolkit/write_file
      name: my_write_tool
      description: Custom write tool from toolkit
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is NOT in shared tools (imported directly to chat)
        self.assertNotIn('my_write_tool', context._tools)

        # Verify tool is accessible from chat by passing the chat config's tools
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('my_write_tool', chat_tools[0].name)
        self.assertEqual('Custom write tool from toolkit', chat_tools[0].description)

    def test_import_single_tool_from_mcp_directly_to_chat(self):
        """Import one tool from MCP server directly in chat section."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'chat_mcp_server': [
                create_mock_mcp_tool('mcp_search', 'Search MCP'),
                create_mock_mcp_tool('mcp_analyze', 'Analyze MCP'),
            ]
        }

        yaml_config = """
mcp:
  chat_mcp_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

chat:
  tools:
    - import_tool: mcp:chat_mcp_server/mcp_search
      name: my_search
      description: Custom search from MCP
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify tool is NOT in shared tools (imported directly to chat)
        self.assertNotIn('my_search', context._tools)

        # Verify tool is accessible from chat by passing the chat config's tools
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('my_search', chat_tools[0].name)
        self.assertEqual('Custom search from MCP', chat_tools[0].description)

    def test_declare_custom_tool_directly_in_chat(self):
        """Declare custom tool directly in chat section."""
        yaml_config = """
chat:
  tools:
    - name: chat_custom_tool
      description: A custom tool in chat
      input:
        - name: query
          description: Search query
          type: str
      body:
        - result: "Chat query result for {query}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool is NOT in shared tools
        self.assertNotIn('chat_custom_tool', context._tools)

        # Verify tool is accessible from chat
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('chat_custom_tool', chat_tools[0].name)
        self.assertEqual('A custom tool in chat', chat_tools[0].description)

    def test_import_toolkit_directly_to_chat(self):
        """Import entire toolkit directly in chat section."""
        yaml_config = """
chat:
  tools:
    - import_tools: llm_workers.tools.fs.FilesystemToolkit
      prefix: 'chat_fs_'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify toolkit tools are NOT in shared tools
        shared_tool_names = list(context._tools.keys())
        self.assertFalse(any(name.startswith('chat_fs_') for name in shared_tool_names))

        # Verify tools are accessible from chat with prefix
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertGreater(len(chat_tools), 0)
        # All tools should have chat_fs_ prefix
        for tool in chat_tools:
            self.assertTrue(tool.name.startswith('chat_fs_'))

    def test_import_mcp_server_directly_to_chat(self):
        """Import all MCP server tools directly in chat section."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'chat_server': [
                create_mock_mcp_tool('chat_read', 'Chat read from MCP'),
                create_mock_mcp_tool('chat_write', 'Chat write to MCP'),
            ]
        }

        yaml_config = """
mcp:
  chat_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

chat:
  tools:
    - import_tools: mcp:chat_server
      prefix: 'direct_'
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify MCP tools are NOT in shared tools
        shared_tool_names = list(context._tools.keys())
        self.assertFalse(any(name.startswith('direct_') for name in shared_tool_names))

        # Verify tools are accessible from chat
        chat_tools = context.get_tools('chat', context.config.chat.tools)
        self.assertEqual(2, len(chat_tools))
        # All tools should have direct_ prefix
        for tool in chat_tools:
            self.assertTrue(tool.name.startswith('direct_'))

    def test_auto_import_mcp_server_directly_to_chat(self):
        """MCP server with auto_import_scope: chat."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'auto_chat_server': [
                create_mock_mcp_tool('auto_tool1', 'Auto tool 1'),
                create_mock_mcp_tool('auto_tool2', 'Auto tool 2'),
            ]
        }

        yaml_config = """
mcp:
  auto_chat_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: chat
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify MCP tools are NOT in shared tools
        shared_tool_names = list(context._tools.keys())
        self.assertFalse(any(name.startswith('auto_chat_server_') for name in shared_tool_names))

        # Verify tools are auto-imported to chat with server name prefix
        chat_tools = context.get_tools('chat', [])
        self.assertEqual(2, len(chat_tools))
        # All tools should have auto_chat_server_ prefix
        tool_names = [tool.name for tool in chat_tools]
        self.assertIn('auto_chat_server_auto_tool1', tool_names)
        self.assertIn('auto_chat_server_auto_tool2', tool_names)


# Test Category 3: Import in Custom Tool Declarations

class TestImportInCustomTools(BaseToolImportTest):
    """Test importing tools within custom tool declarations (via 'call' statements)."""

    def test_import_single_tool_in_custom_tool_call(self):
        """Call statement with import_tool: module.Tool."""
        yaml_config = """
tools:
  - name: wrapper_tool
    description: A tool that wraps read_file
    input:
      - name: filename
        description: File to read
        type: str
    body:
      - call:
          import_tool: llm_workers.tools.fs.ReadFileTool
        params:
          path: "{filename}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify wrapper tool exists
        self.assertIn('wrapper_tool', context._tools)
        tool = context._tools['wrapper_tool']
        self.assertEqual('wrapper_tool', tool.name)

        # Test that the tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['wrapper_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_import_single_tool_from_toolkit_in_custom_tool_call(self):
        """Call statement with toolkit tool reference."""
        yaml_config = """
tools:
  - name: wrapper_tool
    description: A tool that wraps edit_file from toolkit
    input:
      - name: filename
        description: File to edit
        type: str
      - name: content
        description: New content
        type: str
    body:
      - call:
          import_tool: llm_workers.tools.fs.FilesystemToolkit/edit_file
        params:
          path: "{filename}"
          instructions: "{content}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify wrapper tool exists
        self.assertIn('wrapper_tool', context._tools)
        tool = context._tools['wrapper_tool']
        self.assertEqual('wrapper_tool', tool.name)

        # Test that the tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['wrapper_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_import_single_tool_from_mcp_in_custom_tool_call(self):
        """Call statement with MCP tool reference."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'custom_server': [
                create_mock_mcp_tool('mcp_process', 'Process data'),
                create_mock_mcp_tool('mcp_validate', 'Validate data'),
            ]
        }

        yaml_config = """
mcp:
  custom_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - name: wrapper_tool
    description: A tool that wraps MCP process
    input:
      - name: data
        description: Data to process
        type: str
    body:
      - call:
          import_tool: mcp:custom_server/mcp_process
        params:
          input_str: "{data}"
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify wrapper tool exists
        self.assertIn('wrapper_tool', context._tools)
        tool = context._tools['wrapper_tool']
        self.assertEqual('wrapper_tool', tool.name)

        # Test that the tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['wrapper_tool'])
        self.assertEqual(1, len(chat_tools))


# Test Category 4: Import in LLM Tool Declarations

class TestImportInLLMTools(BaseToolImportTest):
    """Test importing tools within LLM tool declarations."""

    def test_import_single_tool_in_llm_tool(self):
        """Import single tool in LLM tool declaration."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_tool
    config:
      tools:
        - import_tool: llm_workers.tools.fs.ReadFileTool
          name: read_tool
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify LLM tool exists
        self.assertIn('llm_with_tool', context._tools)
        llm_tool = context._tools['llm_with_tool']
        self.assertEqual('llm_with_tool', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_import_single_tool_from_toolkit_in_llm_tool(self):
        """Import one tool from toolkit in LLM tool declaration."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_toolkit_tool
    config:
      tools:
        - import_tool: llm_workers.tools.fs.FilesystemToolkit/list_files
          name: list_tool
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify LLM tool exists
        self.assertIn('llm_with_toolkit_tool', context._tools)
        llm_tool = context._tools['llm_with_toolkit_tool']
        self.assertEqual('llm_with_toolkit_tool', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_toolkit_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_import_single_tool_from_mcp_in_llm_tool(self):
        """Import one tool from MCP server in LLM tool declaration."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'llm_server': [
                create_mock_mcp_tool('mcp_query', 'Query data'),
                create_mock_mcp_tool('mcp_fetch', 'Fetch data'),
            ]
        }

        yaml_config = """
mcp:
  llm_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_mcp_tool
    config:
      tools:
        - import_tool: mcp:llm_server/mcp_query
          name: query_tool
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify LLM tool exists
        self.assertIn('llm_with_mcp_tool', context._tools)
        llm_tool = context._tools['llm_with_mcp_tool']
        self.assertEqual('llm_with_mcp_tool', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_mcp_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_declare_custom_tool_in_llm_tool(self):
        """Declare custom tool in LLM tool declaration."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_custom_tool
    config:
      tools:
        - name: custom_inline_tool
          description: A custom tool defined inline
          input:
            - name: input_text
              description: Input text
              type: str
          body:
            - result: "Processed: {input_text}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify LLM tool exists
        self.assertIn('llm_with_custom_tool', context._tools)
        llm_tool = context._tools['llm_with_custom_tool']
        self.assertEqual('llm_with_custom_tool', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_custom_tool'])
        self.assertEqual(1, len(chat_tools))

    def test_import_toolkit_in_llm_tool(self):
        """Import entire toolkit in LLM tool declaration."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_fs_toolkit
    config:
      tools:
        - import_tools: llm_workers.tools.fs.FilesystemToolkit
          prefix: 'fs_'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify LLM tool exists
        self.assertIn('llm_with_fs_toolkit', context._tools)
        llm_tool = context._tools['llm_with_fs_toolkit']
        self.assertEqual('llm_with_fs_toolkit', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_fs_toolkit'])
        self.assertEqual(1, len(chat_tools))

    def test_import_mcp_server_in_llm_tool(self):
        """Import all MCP server tools in LLM tool declaration."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'llm_mcp_server': [
                create_mock_mcp_tool('llm_mcp_tool1', 'LLM MCP tool 1'),
                create_mock_mcp_tool('llm_mcp_tool2', 'LLM MCP tool 2'),
            ]
        }

        yaml_config = """
mcp:
  llm_mcp_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_with_mcp
    config:
      tools:
        - import_tools: mcp:llm_mcp_server
          prefix: 'mcp_'
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify LLM tool exists
        self.assertIn('llm_with_mcp', context._tools)
        llm_tool = context._tools['llm_with_mcp']
        self.assertEqual('llm_with_mcp', llm_tool.name)

        # Verify LLM tool can be retrieved from chat
        chat_tools = context.get_tools('chat', ['llm_with_mcp'])
        self.assertEqual(1, len(chat_tools))


# Test Category 5: Mass Import Properties

class TestMassImportProperties(BaseToolImportTest):
    """Test properties applied to mass-imported tools (filtering, prefixes, UI hints)."""

    def test_toolkit_import_with_tool_filtering_include(self):
        """Test toolkit import with inclusion patterns: tools: ['read*', 'list*']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    filter:
      - 'read*'
      - 'list*'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify only read* and list* tools are imported
        shared_tool_names = list(context._tools.keys())
        self.assertIn('read_file', shared_tool_names)
        self.assertIn('list_files', shared_tool_names)
        self.assertNotIn('write_file', shared_tool_names)
        self.assertNotIn('edit_file', shared_tool_names)

    def test_toolkit_import_with_tool_filtering_exclude(self):
        """Test toolkit import with exclusion patterns: tools: ['*', '!write*']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    filter:
      - '*'
      - '!write*'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify write* tools are excluded
        shared_tool_names = list(context._tools.keys())
        self.assertIn('read_file', shared_tool_names)
        self.assertNotIn('write_file', shared_tool_names)

    def test_toolkit_import_with_tool_filtering_combined(self):
        """Test toolkit import with combined patterns: tools: ['read*', '!read_unsafe']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    filter:
      - 'read*'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify only read* tools are imported
        shared_tool_names = list(context._tools.keys())
        self.assertIn('read_file', shared_tool_names)
        self.assertNotIn('write_file', shared_tool_names)
        self.assertNotIn('list_files', shared_tool_names)

    def test_toolkit_import_with_prefix(self):
        """Test toolkit import with prefix: prefix: 'fs_'."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: 'fs_'
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify all tools have fs_ prefix
        shared_tool_names = list(context._tools.keys())
        self.assertIn('fs_read_file', shared_tool_names)
        self.assertIn('fs_write_file', shared_tool_names)
        self.assertNotIn('read_file', shared_tool_names)  # No unprefixed tools

    def test_toolkit_import_with_empty_prefix(self):
        """Test toolkit import with empty prefix: prefix: ''."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tools have no prefix (original names)
        shared_tool_names = list(context._tools.keys())
        self.assertIn('read_file', shared_tool_names)
        self.assertIn('write_file', shared_tool_names)

    def test_toolkit_import_with_ui_hints_for_all(self):
        """Test toolkit import with ui_hints_for: ['*']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    ui_hints_for: ['*']
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify all tools have UI hints enabled
        for tool_name in context._tools:
            tool = context._tools[tool_name]
            if tool.metadata and 'tool_definition' in tool.metadata:
                tool_def = tool.metadata['tool_definition']
                # UI hint should be True for all tools
                self.assertIsNotNone(tool_def.ui_hint)

    def test_toolkit_import_with_ui_hints_for_pattern(self):
        """Test toolkit import with ui_hints_for: ['read*']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    ui_hints_for: ['read*']
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify only read* tools have UI hints
        read_tool = context._tools.get('read_file')
        if read_tool and read_tool.metadata and 'tool_definition' in read_tool.metadata:
            self.assertIsNotNone(read_tool.metadata['tool_definition'].ui_hint)

    def test_toolkit_import_with_ui_hints_args(self):
        """Test toolkit import with ui_hints_args: ['path']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    ui_hints_for: ['*']
    ui_hints_args: ['path']
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify UI hints args are set
        for tool_name in context._tools:
            tool = context._tools[tool_name]
            if tool.metadata and 'tool_definition' in tool.metadata:
                tool_def = tool.metadata['tool_definition']
                if tool_def.ui_hint:
                    self.assertEqual(['path'], tool_def.ui_hint_args)

    def test_toolkit_import_with_require_confirmation_for_pattern(self):
        """Test toolkit import with require_confirmation_for: ['write*']."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    require_confirmation_for: ['write*']
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify write* tools require confirmation
        write_tool = context._tools.get('write_file')
        if write_tool and write_tool.metadata and 'tool_definition' in write_tool.metadata:
            self.assertTrue(write_tool.metadata['tool_definition'].require_confirmation)

        # Verify read* tools don't require confirmation
        read_tool = context._tools.get('read_file')
        if read_tool and read_tool.metadata and 'tool_definition' in read_tool.metadata:
            self.assertFalse(read_tool.metadata['tool_definition'].require_confirmation)

    def test_mcp_import_with_combined_properties(self):
        """Test MCP import with filtering, prefix, UI hints, and confirmation."""
        # Create mock MCP tools
        mock_mcp_tools = {
            'props_server': [
                create_mock_mcp_tool('read_data', 'Read data from MCP'),
                create_mock_mcp_tool('write_data', 'Write data to MCP'),
                create_mock_mcp_tool('list_data', 'List data from MCP'),
            ]
        }

        yaml_config = """
mcp:
  props_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tools: mcp:props_server
    prefix: 'mcp_'
    filter:
      - '*'
      - '!write*'
    ui_hints_for: ['*']
    ui_hints_args: ['input_str']
    require_confirmation_for: ['list*']
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify filtering: write* excluded
        shared_tool_names = list(context._tools.keys())
        self.assertIn('mcp_read_data', shared_tool_names)
        self.assertIn('mcp_list_data', shared_tool_names)
        self.assertNotIn('mcp_write_data', shared_tool_names)

        # Verify prefix applied
        self.assertTrue(all(name.startswith('mcp_') for name in shared_tool_names))

        # Verify UI hints and confirmation for matching tools
        list_tool = context._tools.get('mcp_list_data')
        if list_tool and list_tool.metadata and 'tool_definition' in list_tool.metadata:
            tool_def = list_tool.metadata['tool_definition']
            self.assertIsNotNone(tool_def.ui_hint)
            self.assertTrue(tool_def.require_confirmation)


# Test Category 6: Single Import Properties

class TestSingleImportProperties(BaseToolImportTest):
    """Test properties applied to single-imported tools."""

    def test_single_tool_import_with_name_override(self):
        """Test single tool import with name: custom_name."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: my_read_tool
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify tool name is overridden
        self.assertIn('my_read_tool', context._tools)
        self.assertNotIn('read_file', context._tools)

        tool = context._tools['my_read_tool']
        self.assertEqual('my_read_tool', tool.name)

    def test_single_tool_import_with_description_override(self):
        """Test single tool import with description: Custom desc."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    description: Custom description for reading files
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify description is overridden
        tool = context._tools['read_tool']
        self.assertEqual('Custom description for reading files', tool.description)

    def test_single_tool_import_with_ui_hint_string(self):
        """Test single tool import with ui_hint: 'Doing {action}'."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    ui_hint: "Reading file ${path}"
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify UI hint is set as string
        tool = context._tools['read_tool']
        tool_def = tool.metadata['tool_definition']
        self.assertEqual("Reading file file", tool_def.ui_hint.evaluate({'path': 'file'}))

    def test_single_tool_import_with_ui_hint_boolean(self):
        """Test single tool import with ui_hint: true or false."""
        # Test with ui_hint: true
        yaml_config_true = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool_true
    ui_hint: true
"""
        context = self.create_and_initialize_context(yaml_config_true)
        tool = context._tools['read_tool_true']
        tool_def = tool.metadata['tool_definition']
        self.assertEqual(True, tool_def.ui_hint)

        # Test with ui_hint: false
        yaml_config_false = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool_false
    ui_hint: false
"""
        context = self.create_and_initialize_context(yaml_config_false)
        tool = context._tools['read_tool_false']
        tool_def = tool.metadata['tool_definition']
        self.assertEqual(False, tool_def.ui_hint)

    def test_single_tool_import_with_require_confirmation(self):
        """Test single tool import with require_confirmation: true."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    require_confirmation: true
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify require_confirmation is set
        tool = context._tools['read_tool']
        tool_def = tool.metadata['tool_definition']
        self.assertTrue(tool_def.require_confirmation)

    def test_single_tool_import_with_return_direct(self):
        """Test single tool import with return_direct: true."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    return_direct: true
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify return_direct is set
        tool = context._tools['read_tool']
        self.assertTrue(tool.return_direct)

    def test_single_tool_import_with_confidential(self):
        """Test single tool import with confidential: true (implies return_direct)."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_tool
    confidential: true
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify confidential implies return_direct
        tool = context._tools['read_tool']
        tool_def = tool.metadata['tool_definition']
        self.assertTrue(tool_def.confidential)
        self.assertTrue(tool.return_direct)  # Should be automatically set


# Test Category 7: Shared Tool Referencing

class TestSharedToolReferencing(BaseToolImportTest):
    """Test pattern matching when referencing shared tools."""

    def test_reference_shared_tool_by_exact_name(self):
        """Test referencing shared tool by exact name: tools: ['search_web']."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_file
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: write_file
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify exact name match works
        chat_tools = context.get_tools('chat', ['read_file'])
        self.assertEqual(1, len(chat_tools))
        self.assertEqual('read_file', chat_tools[0].name)

    def test_reference_shared_tool_by_wildcard_pattern(self):
        """Test referencing shared tool by wildcard: tools: ['search_*']."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: search_files
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: search_text
  - import_tool: llm_workers.tools.fs.EditFileTool
    name: edit_file
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify wildcard pattern matches both search_ tools
        chat_tools = context.get_tools('chat', [ToolsReference(match=['search_*'])])
        tool_names = [tool.name for tool in chat_tools]
        self.assertIn('search_files', tool_names)
        self.assertIn('search_text', tool_names)
        self.assertNotIn('edit_file', tool_names)

    def test_reference_shared_tool_by_multiple_patterns(self):
        """Test referencing shared tools by multiple patterns: tools: ['search_*', 'read_*']."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: search_files
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: read_files
  - import_tool: llm_workers.tools.fs.EditFileTool
    name: edit_file
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify multiple patterns work
        chat_tools = context.get_tools('chat', [ToolsReference(match=['search_*', 'read_*'])])
        tool_names = [tool.name for tool in chat_tools]
        self.assertIn('search_files', tool_names)
        self.assertIn('read_files', tool_names)
        self.assertNotIn('edit_file', tool_names)

    def test_reference_shared_tool_with_exclusion_pattern(self):
        """Test referencing shared tools with exclusion: tools: ['*', '!write_*']."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_file
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: write_file
  - import_tool: llm_workers.tools.fs.EditFileTool
    name: edit_file
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify exclusion pattern works
        chat_tools = context.get_tools('chat', [ToolsReference(match=['*', '!write_*'])])
        tool_names = [tool.name for tool in chat_tools]
        self.assertIn('read_file', tool_names)
        self.assertIn('edit_file', tool_names)
        self.assertNotIn('write_file', tool_names)

    def test_reference_nonexistent_shared_tool_fails(self):
        """Test that referencing non-existent shared tool raises error."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: read_file
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: write_file
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify that referencing non-existent tool by exact name raises error
        with self.assertRaises(Exception):
            context.get_tools('chat', ['nonexistent_tool'])

        # Verify that referencing with ToolsReference pattern that matches 0 tools raises error
        with self.assertRaises(Exception):
            context.get_tools('chat', [ToolsReference(match=['nonexistent_*'])])


# Test Category 8: Error Handling

class TestToolImportingErrors(BaseToolImportTest):
    """Test error handling and edge cases in tool importing."""

    def test_import_nonexistent_tool_fails(self):
        """Test that importing non-existent tool raises WorkerException."""
        yaml_config = """
tools:
  - import_tool: nonexistent.module.NonExistentTool
    name: fake_tool
"""
        # Verify that importing non-existent tool raises error
        with self.assertRaises(Exception):
            self.create_and_initialize_context(yaml_config)

    def test_import_nonexistent_toolkit_fails(self):
        """Test that importing non-existent toolkit raises WorkerException."""
        yaml_config = """
tools:
  - import_tools: nonexistent.toolkit.FakeToolkit
    prefix: 'fake_'
"""
        # Verify that importing non-existent toolkit raises error
        with self.assertRaises(Exception):
            self.create_and_initialize_context(yaml_config)

    def test_import_nonexistent_mcp_server_fails(self):
        """Test that importing from unconfigured MCP server raises WorkerException."""
        yaml_config = """
tools:
  - import_tools: mcp:nonexistent_server
    prefix: 'mcp_'
"""
        # Verify that importing from unconfigured MCP server raises error
        with self.assertRaises(Exception):
            self.create_and_initialize_context(yaml_config)

    def test_duplicate_tool_name_fails(self):
        """Test that importing tool with duplicate name raises WorkerException."""
        yaml_config = """
tools:
  - import_tool: llm_workers.tools.fs.ReadFileTool
    name: my_tool
  - import_tool: llm_workers.tools.fs.WriteFileTool
    name: my_tool
"""
        # Verify that duplicate tool names raise error
        with self.assertRaises(Exception) as context:
            self.create_and_initialize_context(yaml_config)

        # Verify the error message mentions the duplicate tool
        self.assertIn('my_tool', str(context.exception))
        self.assertIn('already defined', str(context.exception))

    def test_empty_pattern_matches_nothing(self):
        """Test that empty tool filter patterns match no tools."""
        yaml_config = """
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: ''
    filter: []
"""
        context = self.create_and_initialize_context(yaml_config)

        # Verify that no tools are imported with empty pattern
        shared_tool_names = list(context._tools.keys())
        self.assertEqual(0, len(shared_tool_names))

    def test_exclusion_only_with_implicit_wildcard(self):
        """Test that tools: ['!unsafe'] matches all except unsafe (implicit '*')."""
        # Create mock MCP tools including an "unsafe" tool
        mock_mcp_tools = {
            'test_server': [
                create_mock_mcp_tool('safe_tool', 'Safe tool'),
                create_mock_mcp_tool('unsafe_tool', 'Unsafe tool'),
                create_mock_mcp_tool('another_safe', 'Another safe tool'),
            ]
        }

        yaml_config = """
mcp:
  test_server:
    transport: stdio
    command: test_command
    args: []
    auto_import_scope: none

tools:
  - import_tools: mcp:test_server
    prefix: ''
    filter:
      - '!unsafe*'
"""
        context = self.create_and_initialize_context(yaml_config, mock_mcp_tools)

        # Verify that exclusion-only pattern matches all except unsafe*
        shared_tool_names = list(context._tools.keys())
        self.assertIn('safe_tool', shared_tool_names)
        self.assertIn('another_safe', shared_tool_names)
        self.assertNotIn('unsafe_tool', shared_tool_names)


if __name__ == '__main__':
    unittest.main()
