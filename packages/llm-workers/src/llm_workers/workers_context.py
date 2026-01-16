import asyncio
import importlib
import inspect
import logging
from asyncio import AbstractEventLoop
from contextlib import AsyncExitStack
from copy import copy
from typing import Dict, List, Optional, Callable, Any

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from llm_workers.api import WorkersContext, WorkerException, ExtendedBaseTool, UserContext
from llm_workers.config import WorkersConfig, ImportToolStatement, ImportToolsStatement, \
    ToolDefinition, CustomToolDefinition, ToolsDefinitionStatement, MCPServerStdio, MCPServerHttp, \
    ToolsDefinitionOrReference, ToolsReference
from llm_workers.expressions import EvaluationContext
from llm_workers.utils import matches_patterns, load_yaml

logger = logging.getLogger(__name__)


class StandardWorkersContext(WorkersContext):

    _tools: Dict[str, BaseTool] = {}
    _loop: Optional[AbstractEventLoop] = None
    _mcp_client: Optional[MultiServerMCPClient] = None
    _mcp_sessions: Dict[str, dict] = {}
    _mcp_tools_by_server: dict[str, list[BaseTool]]

    def __init__(self, config: WorkersConfig, user_context: UserContext):
        self._config = config
        self._user_context = user_context
        self._tools: Dict[str, BaseTool] = {}
        self._mcp_tools_by_server: dict[str, list[BaseTool]] = {}
        self._evaluation_context = EvaluationContext({"env": user_context.environment.copy()})

    @classmethod
    def load_script(cls, name: str) -> WorkersConfig:
        logger.info(f"Loading {name}")
        return WorkersConfig(**(load_yaml(name)))

    @property
    def config(self) -> WorkersConfig:
        return self._config

    @property
    def evaluation_context(self) -> EvaluationContext:
        return self._evaluation_context

    @property
    def shared_tools(self) -> Dict[str, BaseTool]:
        return self._tools

    def get_tool(self, tool_ref: str, extra_tools: Optional[Dict[str, BaseTool]] = None) -> BaseTool:
        if extra_tools and tool_ref in extra_tools:
            return extra_tools[tool_ref]
        if tool_ref in self._tools:
            return self._tools[tool_ref]
        else:
            available_tools = list(self._tools.keys())
            if extra_tools:
                available_tools.extend(extra_tools.keys())
            available_tools.sort()
            raise ValueError(f"Tool {tool_ref} not found, available tools: {available_tools}")

    def get_tools(self, scope: str, tool_refs: List[ToolsDefinitionOrReference]) -> List[BaseTool]:
        results: list[BaseTool] = []
        create_tool_statements: List[ToolsDefinitionOrReference] = []
        for tool_ref in tool_refs:
            if isinstance(tool_ref, str):
                results.append(self.get_tool(tool_ref))
            elif isinstance(tool_ref, ToolsReference):
                matches = 0
                for tool in self._tools.values():
                    if not matches_patterns(tool.name, tool_ref.match):
                        continue
                    results.append(tool)
                    matches += 1
                if matches == 0:
                    raise ValueError(f"No tools matched patterns: {tool_ref}")
            else:
                create_tool_statements.append(tool_ref)
        # Now create tools for all definitions (call even if empty for auto-import)
        new_tools: dict[str, BaseTool] = {}
        self._create_tools(scope, new_tools, create_tool_statements)
        results.extend(new_tools.values())
        return results

    def get_llm(self, llm_name: str):
        return self._user_context.get_llm(llm_name)

    def run(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Properly initializes this context, then runs provided function with parameters and shuts down the context.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._run(func, *args, **kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _run(self, func: Callable[..., Any], *args, **kwargs):
        async with AsyncExitStack() as stack:
            self._loop = asyncio.get_running_loop() # capture for sync wrappers

            if len(self._config.mcp) > 0:
                logger.info("Initializing MCP clients...")
                mcp_client = MultiServerMCPClient(self._build_server_configs())
                for server_name in self._config.mcp.keys():
                    try:
                        logger.info(f"Connecting to MCP server '{server_name}'...")
                        session = await stack.enter_async_context(mcp_client.session(server_name))
                        self._mcp_tools_by_server[server_name] = await self._load_mcp_tools(server_name, session)
                    except Exception as e:
                        raise WorkerException(f"Failed to load tools from MCP server '{server_name}': {e}")

            # resolve and expose "global data"
            for key, expr in self._config.shared.data.items():
                self._evaluation_context.add(key, expr.evaluate(self._evaluation_context))
            # lock the evaluation context to prevent further modifications
            self._evaluation_context.mutable = False

            # finally we can register all tools
            self._create_tools('shared', self._tools, self._config.shared.tools)

            # and run the business code
            return await asyncio.to_thread(func, *args, **kwargs)

    def _build_server_configs(self) -> Dict[str, dict]:
        server_configs = {}
        for server_name, server_def in self._config.mcp.items():
            if isinstance(server_def, MCPServerStdio):
                server_configs[server_name] = {
                    "transport": "stdio",
                    "command": server_def.command,
                    "args": server_def.args.evaluate(self._evaluation_context),
                    "env": server_def.env.evaluate(self._evaluation_context),
                }
            elif isinstance(server_def, MCPServerHttp):
                server_configs[server_name] = {
                    "transport": "streamable_http",
                    "url": server_def.url,
                    "headers": server_def.headers.evaluate(self._evaluation_context),
                }
            else:
                raise WorkerException(f"Unsupported MCP definition: {type(server_def)}")
        return server_configs

    async def _load_mcp_tools(self, server_name: str, session) -> list[BaseTool]:
        """Load tools from a single MCP server session."""
        server_tools = await load_mcp_tools(session)

        # Tag each tool with its server and add sync wrapper
        for tool in server_tools:
            if tool.metadata is None:
                tool.metadata = {}
            tool.metadata['mcp_server'] = server_name
            tool.metadata['original_name'] = tool.name

            # Add synchronous func wrapper using persistent event loop
            if hasattr(tool, 'coroutine') and tool.coroutine is not None and tool.func is None:
                tool.func = self._make_sync_wrapper(tool.coroutine)
                logger.debug(f"Added sync wrapper to MCP tool '{tool.name}'")

        logger.info(f"Loaded {len(server_tools)} tools from MCP server '{server_name}'")
        return server_tools

    def _make_sync_wrapper(self, async_func):
        """
        Create a sync wrapper for an async function that uses the persistent event loop.

        CRITICAL: Must use self._loop instead of asyncio.run() because MCP tools
        require the same event loop and session context they were created in.
        """
        loop = self._loop  # Capture loop reference

        def sync_wrapper(*args, **kwargs):
            if loop is None or loop.is_closed():
                raise RuntimeError("StandardWorkersContext has been closed")
            future = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), loop)
            return future.result()

        return sync_wrapper

    def _import_tools_from_statement(self, scope: str, results: Dict[str, BaseTool], import_stmt: ImportToolsStatement):
        """Import tools based on ImportToolsStatement, delegating to toolkit or MCP import."""
        (schema, path) = import_stmt.import_tools_split
        if schema == '':
            StandardWorkersContext._import_toolkit_tools(scope, results, path, import_stmt)
        elif schema == 'mcp':
            self._import_mcp_tools(scope, results, path, import_stmt)
        else:
            raise WorkerException(f"Unsupported tool import schema '{schema}' in {scope}")

    def _create_tools(self, scope: str, results: Dict[str, BaseTool], statements: List[ToolsDefinitionStatement]):
        # First auto-register MCP tools if any - they might be referenced from tools
        self._auto_register_mcp_tools(scope, results)
        # Then process all tool statements
        for tool_entry in statements:
            if isinstance(tool_entry, ImportToolsStatement):
                self._import_tools_from_statement(scope, results, tool_entry)
            else:
                # Regular ToolDefinition
                tool = self._create_tool(tool_entry)
                if tool.name in results:
                    raise WorkerException(f"Cannot register tool {tool.name} in {scope}: tool already defined")
                results[tool.name] = tool
                logger.info(f"Registered tool '{tool.name}' in {scope}")

    def _create_tool(self, tool_def: ToolDefinition) -> BaseTool:
        from llm_workers.tools.custom_tool import build_custom_tool
        try:
            if isinstance(tool_def, ImportToolStatement):
                tool = self._import_tool(tool_def)
            elif isinstance(tool_def, CustomToolDefinition):
                tool = build_custom_tool(tool_def, self)
            else:
                raise WorkerException(f"Unsupported tool definition type: {type(tool_def)}")
            # common post-processing
            if tool_def.return_direct is not None:
                tool.return_direct = tool_def.return_direct
            if tool_def.confidential:   # confidential implies return_direct
                tool.return_direct = True
            if tool.metadata is None:
                tool.metadata = {}
            tool.metadata['tool_definition'] = tool_def
            if isinstance(tool, ExtendedBaseTool):
                tool.metadata['__extension'] = tool # really hackish
            return tool
        except ImportError as e:
            raise WorkerException(f"Failed to import module for tool {tool_def}: {e}")
        except WorkerException:
            raise
        except Exception as e:
            raise WorkerException(f"Failed to create tool {tool_def}: {e}", e)

    def _import_tool(self, tool_def: ImportToolStatement) -> BaseTool:
        # Check if import_tool contains '/' for toolkit/tool_name or mcp:server/tool_name syntax
        if '/' in tool_def.import_tool:
            # Split into path and tool name
            toolkit_path, tool_name = tool_def.import_tool.rsplit('/', 1)

            # Create a fake ImportToolsStatement with filter for single tool
            fake_import_stmt = ImportToolsStatement(
                import_tools=toolkit_path,
                prefix='',
                filter=[tool_name],
            )

            # Use a temporary results dict to collect the tool
            temp_results: Dict[str, BaseTool] = {}

            # Delegate to _import_tools_from_statement
            self._import_tools_from_statement('temp', temp_results, fake_import_stmt)

            # Should have exactly one tool
            if len(temp_results) != 1:
                available = list(temp_results.keys())
                raise WorkerException(
                    f"Expected exactly one tool '{tool_name}' from {toolkit_path}, "
                    f"but got {len(temp_results)} tools: {available}"
                )

            # Extract the single tool
            tool = list(temp_results.values())[0]

            # Override properties from tool_def (similar to existing logic below)
            if tool_def.name is not None:
                tool.name = tool_def.name
            if tool_def.description is not None:
                tool.description = tool_def.description

            # Update tool_definition in metadata with additional properties
            if tool.metadata is None:
                tool.metadata = {}

            tool_def = tool_def.model_copy()
            tool_def.name = tool.name
            tool.metadata['tool_definition'] = tool_def

            return tool

        # Original logic for direct tool imports
        tool_config = copy(tool_def.config if tool_def.config else {})
        # split model.import_tool into module_name and symbol
        segments = tool_def.import_tool.split('.')
        module_name = '.'.join(segments[:-1])
        symbol_name = segments[-1]
        module = importlib.import_module(module_name)  # Import the module
        symbol = getattr(module, symbol_name)  # Retrieve the symbol
        # make the tool
        if symbol is None:
            raise ValueError(f"Cannot import tool from {tool_def.import_tool}: symbol {symbol_name} not found")
        elif isinstance(symbol, BaseTool):
            tool = symbol
        elif inspect.isclass(symbol):
            # For class constructors, add name/description to tool_config
            if tool_def.name is not None:
                tool_config['name'] = tool_def.name
            if tool_def.description is not None:
                tool_config['description'] = tool_def.description
            tool = symbol(**tool_config) # use default constructor
        elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
            # For factory functions, DON'T add name/description to tool_config
            # They will be set after the tool is created
            if len(symbol.__annotations__) >= 2 and 'context' in symbol.__annotations__ and 'tool_config' in symbol.__annotations__:
                tool = symbol(context = self, tool_config = tool_config)
            else:
                raise ValueError("Invalid tool factory signature, must be `def factory(context: WorkersContext, tool_config: dict[str, any]) -> BaseTool`")
        else:
            raise ValueError(f"Invalid symbol type {type(symbol)}")
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Not a BaseTool: {type(tool)}")
        # overrides for un-cooperating tools (and factories)
        if tool_def.name is not None:
            tool.name = tool_def.name
        if tool_def.description is not None:
            tool.description = tool_def.description
        return tool

    @staticmethod
    def _import_toolkit_tools(scope: str, results: Dict[str, BaseTool], path: str, import_def: ImportToolsStatement):
        """Import and register tools from a BaseToolkit."""
        try:
            segments = path.split('.')
            module_name = '.'.join(segments[:-1])
            class_name = segments[-1]
            module = importlib.import_module(module_name)
            toolkit_class = getattr(module, class_name)

            if not inspect.isclass(toolkit_class) or not issubclass(toolkit_class, BaseToolkit):
                raise WorkerException(f"Not a BaseToolkit: {path}")

            toolkit = toolkit_class()
            tools = toolkit.get_tools()

            logger.debug(f"Loaded {len(tools)} tools from toolkit {path}")

            StandardWorkersContext._import_tools(
                scope,
                results,
                tools,
                import_def.prefix,
                import_def.filter,
                import_def.force_ui_hints_for,
                import_def.force_no_ui_hints_for,
                import_def.ui_hints_args,
                import_def.force_require_confirmation_for,
                import_def.force_no_confirmation_for,
                origin=f"toolkit {path}")

        except ImportError as e:
            raise WorkerException(f"Failed to import toolkit {import_def.import_toolkit} for {scope}: {e}")
        except Exception as e:
            raise WorkerException(f"Failed to register toolkit tools from {import_def.import_toolkit} for {scope}: {e}", e)

    def _import_mcp_tools(self, scope: str, results: Dict[str, BaseTool], path: str, import_def: ImportToolsStatement):
            if path not in self._mcp_tools_by_server:
                raise WorkerException(f"MCP server '{path}' not found for tool import")

            self._import_tools(
                scope,
                results,
                self._mcp_tools_by_server[path],
                import_def.prefix,
                import_def.filter,
                import_def.force_ui_hints_for,
                import_def.force_no_ui_hints_for,
                import_def.ui_hints_args,
                import_def.force_require_confirmation_for,
                import_def.force_no_confirmation_for,
                origin=f"MCP server '{path}'")

    def _auto_register_mcp_tools(self, scope: str, results: Dict[str, BaseTool]):
        """Register MCP tools from servers with import_tools_to_chat=True."""
        for server_name, tools in self._mcp_tools_by_server.items():
            server_def = self._config.mcp.get(server_name)

            if server_def.auto_import_scope != scope:
                logger.debug(f"Skipping auto-import for MCP server '{server_name}' in {scope}")
                continue

            self._import_tools(
                scope,
                results,
                tools,
                tools_prefix= f"{server_name}_",
                tools_filters=['*'],
                force_ui_hints_for=['*'],
                force_no_ui_hints_for=[],
                ui_hint_args=[],
                force_require_confirmation_for=[],
                force_no_confirmation_for=[],
                origin=f"MCP server '{server_name}'")

    @staticmethod
    def _import_tools(scope: str, results: Dict[str, BaseTool], tools: List[BaseTool], tools_prefix: str,
                      tools_filters: list[str], force_ui_hints_for: list[str], force_no_ui_hints_for: list[str],
                      ui_hint_args: list[str], force_require_confirmation_for: list[str],
                      force_no_confirmation_for: list[str],
                      origin: str):
        registered = 0
        for tool in tools:
            original_name = tool.name

            if not matches_patterns(original_name, tools_filters):
                logger.debug(f"Skipping tool '{original_name}' from {origin} (filtered by patterns)")
                continue

            prefixed_name = f"{tools_prefix}{original_name}"

            if prefixed_name in results:
                raise WorkerException(f"Cannot register tool {prefixed_name} in {scope}: tool already defined")

            # Determine ui_hint value
            ui_hint = None  # Default: let tool's built-in behavior apply
            if matches_patterns(original_name, force_ui_hints_for):
                ui_hint = True  # Force enable
            elif matches_patterns(original_name, force_no_ui_hints_for):
                ui_hint = False  # Force disable
            # else: None → fallback to tool's ExtendedBaseTool.get_ui_hint()

            # Determine require_confirmation value
            require_confirm = None  # Default: let tool's built-in behavior apply
            if matches_patterns(original_name, force_require_confirmation_for):
                require_confirm = True  # Force enable
            elif matches_patterns(original_name, force_no_confirmation_for):
                require_confirm = False  # Force disable
            # else: None → fallback to tool's ExtendedBaseTool.needs_confirmation()

            tool_def = ToolDefinition(
                name=prefixed_name,
                description=tool.description,
                ui_hint=ui_hint,
                ui_hint_args=ui_hint_args if ui_hint is not False else [],
                require_confirmation=require_confirm,
            )

            # Apply settings and register
            tool.name = prefixed_name
            tool.metadata = tool.metadata or {}
            tool.metadata['tool_definition'] = tool_def
            tool.metadata['original_name'] = original_name
            if isinstance(tool, ExtendedBaseTool):
                tool.metadata['__extension'] = tool # really hackish

            results[prefixed_name] = tool
            logger.info(f"Registered tool '{prefixed_name}' from {origin} in {scope}")
            registered += 1

        logger.info(f"Registered {registered}/{len(tools)} tools from {origin} in {scope}")
