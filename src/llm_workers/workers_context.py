import asyncio
import importlib
import inspect
import logging
from asyncio import AbstractEventLoop
from copy import copy
from typing import Dict, List, Optional

import yaml
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from llm_workers.api import WorkersContext, WorkerException, ExtendedBaseTool, UserContext
from llm_workers.config import WorkersConfig, ToolDefinition, ToolReference
from llm_workers.tools.custom_tool import build_custom_tool
from llm_workers.utils import matches_patterns, substitute_env_vars_in_list, substitute_env_vars_in_dict

logger = logging.getLogger(__name__)


class StandardWorkersContext(WorkersContext):

    _tools: Dict[str, BaseTool] = {}
    _loop: Optional[AbstractEventLoop] = None
    _mcp_client: Optional[MultiServerMCPClient] = None
    _mcp_sessions: Dict[str, dict] = {}

    def __init__(self, config: WorkersConfig, user_context: UserContext):
        self._config = config
        self._user_context = user_context
        self._register_tools()

    def _register_tools(self):
        for tool_def in self._config.tools:
            if tool_def.name in self._tools:
                raise WorkerException(f"Failed to create tool {tool_def.name}: tool already defined")
            tool = self._create_tool(tool_def)
            self._tools[tool.name] = tool
            logger.info(f"Registered tool {tool.name}")

    def _create_tool(self, tool_def: ToolDefinition) -> BaseTool:
        try:
            if tool_def.import_from is not None:
                tool = self._import_tool(tool_def)
            else:
                tool = build_custom_tool(tool_def, self)
            # common post-processing
            if tool_def.return_direct is not None:
                tool.return_direct = tool_def.return_direct
            if tool_def.confidential:   # confidential implies return_direct
                tool.return_direct = True
            if tool.metadata is None:
                tool.metadata = {}
            tool.metadata['tool_definition'] = tool_def
            if isinstance(tool, ExtendedBaseTool):
                tool.metadata['__extension'] = tool # TODO really hackish
            return tool
        except ImportError as e:
            raise WorkerException(f"Failed to import module for tool {tool_def.name}: {e}")
        except Exception as e:
            raise WorkerException(f"Failed to create tool {tool_def.name}: {e}", e)

    def _import_tool(self, tool_def: ToolDefinition) -> BaseTool:
        tool_config = copy(tool_def.config if tool_def.config else tool_def.model_extra)
        tool_config['name'] = tool_def.name
        if tool_def.description is not None:
            tool_config['description'] = tool_def.description
        # split model.import_from into module_name and symbol
        segments = tool_def.import_from.split('.')
        module_name = '.'.join(segments[:-1])
        symbol_name = segments[-1]
        module = importlib.import_module(module_name)  # Import the module
        symbol = getattr(module, symbol_name)  # Retrieve the symbol
        # make the tool
        if symbol is None:
            raise ValueError(f"Cannot import tool from {tool_def.import_from}: symbol {symbol_name} not found")
        elif isinstance(symbol, BaseTool):
            tool = symbol
        elif inspect.isclass(symbol):
            tool = symbol(**tool_config) # use default constructor
        elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
            if len(symbol.__annotations__) >= 2 and 'context' in symbol.__annotations__ and 'tool_config' in symbol.__annotations__:
                tool = symbol(context = self, tool_config = tool_config)
            else:
                raise ValueError("Invalid tool factory signature, must be `def factory(context: WorkersContext, tool_config: dict[str, any]) -> BaseTool`")
        else:
            raise ValueError(f"Invalid symbol type {type(symbol)}")
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Not a BaseTool: {type(tool)}")
        # overrides for un-cooperating tools
        tool.name = tool_def.name
        if tool_def.description is not None:
            tool.description = tool_def.description
        return tool

    @classmethod
    def load_script(cls, name: str) -> WorkersConfig:
        logger.info(f"Loading {name}")
        # if name has module:resource format, load it as a module
        if ':' in name:
            module, resource = name.split(':', 1)
            if len(module) > 1: # ignore volume names on windows
                with importlib.resources.files(module).joinpath(resource).open("r") as file:
                    config_data = yaml.safe_load(file)
                return WorkersConfig(**config_data)
        # try loading as file
        with open(name, 'r') as file:
            config_data = yaml.safe_load(file)
        return WorkersConfig(**config_data)

    @property
    def config(self) -> WorkersConfig:
        return self._config

    @property
    def get_public_tools(self) -> List[BaseTool]:
        public_tools = []
        for tool in self._tools.values():
            if not tool.name.startswith("_"):
                public_tools.append(tool)
        return public_tools

    def _register_tool(self, tool: BaseTool):
        redefine = tool.name in self._tools
        self._tools[tool.name] = tool
        if redefine:
            logger.info(f"Redefined tool {tool.name}")
        else:
            logger.info(f"Registered tool {tool.name}")

    def get_tool(self, tool_ref: ToolReference) -> BaseTool:
        if isinstance(tool_ref, ToolDefinition):
            return self._create_tool(tool_ref)
        if tool_ref in self._tools:
            return self._tools[tool_ref]
        else:
            available_tools = list(self._tools.keys())
            available_tools.sort()
            raise ValueError(f"Tool {tool_ref} not found, available tools: {available_tools}")

    def get_llm(self, llm_name: str):
        return self._user_context.get_llm(llm_name)

    async def __aenter__(self):
        """
        Initialize MCP clients and load tools asynchronously.
        """
        self._loop = asyncio.get_running_loop()

        if self._config.mcp is None or len(self._config.mcp) == 0:
            return self

        logger.info("Initializing MCP clients...")

        # Build server configs
        server_configs = {}
        for server_name, server_def in self._config.mcp.items():
            try:
                if server_def.transport == "stdio":
                    server_configs[server_name] = {
                        "transport": "stdio",
                        "command": server_def.command,
                        "args": substitute_env_vars_in_list(server_def.args),
                        "env": substitute_env_vars_in_dict(server_def.env),
                    }
                elif server_def.transport == "streamable_http":
                    server_configs[server_name] = {
                        "transport": "streamable_http",
                        "url": server_def.url,
                        "headers": substitute_env_vars_in_dict(server_def.headers),
                    }
                else:
                    raise RuntimeError(f"Unsupported MCP transport: {server_def.transport}")
                logger.info(f"Configured MCP server '{server_name}'")
            except Exception as e:
                logger.error(f"Failed to configure MCP server '{server_name}': {e}", exc_info=True)
                # Continue with other servers

        if not server_configs:
            logger.warning("No valid MCP server configurations found")
            return self

        # Load tools from MCP servers
        try:
            tools_by_server = await self._load_mcp_tools_async(server_configs)
            self._register_mcp_tools(tools_by_server)
        except Exception as e:
            logger.error(f"Failed to initialize MCP clients: {e}", exc_info=True)
            # Don't raise - allow the system to continue with regular tools

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Close this context (including all MCP server) sessions and cleanup.
        """
        if not self._mcp_sessions:
            return

        logger.info(f"Closing {len(self._mcp_sessions)} MCP sessions...")

        # Close each session by calling __aexit__ on stored context managers
        for server_name, session_data in self._mcp_sessions.items():
            try:
                logger.debug(f"Closing MCP session for '{server_name}'")
                context_manager = session_data['context_manager']
                # Call __aexit__ with no exception info (None, None, None)
                await context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Failed to close MCP session for '{server_name}': {e}", exc_info=True)

        self._mcp_sessions.clear()
        logger.info("MCP sessions closed")

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

    async def _load_mcp_tools_async(self, server_configs: Dict[str, dict]) -> Dict[str, List[BaseTool]]:
        """Load tools from MCP servers asynchronously, keeping sessions open."""

        # Create and store client
        self._mcp_client = MultiServerMCPClient(server_configs)
        tools_by_server = {}

        # Load tools from each server individually and keep sessions open
        for server_name in server_configs.keys():
            try:
                logger.info(f"Connecting to MCP server '{server_name}'...")

                # Enter session but don't exit - store the context manager
                session_cm = self._mcp_client.session(server_name)
                session = await session_cm.__aenter__()

                # Store session for later cleanup
                self._mcp_sessions[server_name] = {
                    'session': session,
                    'context_manager': session_cm
                }

                # Load tools from the open session
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

                tools_by_server[server_name] = server_tools
                logger.info(f"Loaded {len(server_tools)} tools from MCP server '{server_name}'")
            except Exception as e:
                logger.error(f"Failed to load tools from MCP server '{server_name}': {e}", exc_info=True)
                tools_by_server[server_name] = []  # Empty list for failed servers

        return tools_by_server

    def _register_mcp_tools(self, tools_by_server: Dict[str, List[BaseTool]]):
        """Filter and register MCP tools based on configuration."""
        for server_name, tools in tools_by_server.items():
            server_def = self._config.mcp.get(server_name)
            if server_def is None:
                continue

            for tool in tools:
                original_name = tool.metadata.get('original_name', tool.name)

                # Check if tool matches filter patterns
                if not matches_patterns(original_name, server_def.tools):
                    logger.debug(f"Skipping MCP tool '{original_name}' from server '{server_name}' (filtered by patterns)")
                    continue

                # Create prefixed tool name
                prefixed_name = f"{server_name}_{original_name}"

                # Check for name conflicts
                if prefixed_name in self._tools:
                    logger.warning(f"MCP tool name conflict: '{prefixed_name}' already exists, skipping")
                    continue

                # Determine UI hint and arguments display
                ui_hint = None
                ui_hint_args = None
                if matches_patterns(original_name, server_def.ui_hints_for):
                    ui_hint = True
                    ui_hint_args = server_def.ui_hints_args

                # Determine if confirmation is required
                require_confirmation = matches_patterns(original_name, server_def.require_confirmation_for)

                # Create ToolDefinition for MCP tool
                tool_def = ToolDefinition(
                    name=prefixed_name,
                    description=tool.description,
                    ui_hint=ui_hint,
                    ui_hint_args=ui_hint_args,
                    require_confirmation=require_confirmation,
                )

                # Update tool metadata
                tool.metadata['tool_definition'] = tool_def

                # Override tool name with prefixed version
                tool.name = prefixed_name

                # Register tool
                self._register_tool(tool)

            # Summary log for this server
            registered_count = len([t for t in tools if t.name.startswith(f"{server_name}_")])
            logger.info(f"Registered {registered_count}/{len(tools)} tools from MCP server '{server_name}'")

