"""Agent access to quantowl: a dependency-free MCP server over stdio.

Start the server with ``quantowl-mcp`` (installed by ``pip install -e .``) or
``python -m quantowl.agent``. Any MCP client can then call the tools in
:mod:`quantowl.agent.tools`. Each tool takes JSON arguments and returns compact JSON.

The server speaks JSON-RPC 2.0, one message per line, using MCP protocol version
2024-11-05. It needs no SDK, so it adds no dependency.
"""

from .server import handle_message, main
from .tools import TOOLS, call_tool

__all__ = ["handle_message", "main", "TOOLS", "call_tool"]
