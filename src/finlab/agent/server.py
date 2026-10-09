"""Minimal MCP server over stdio (JSON-RPC 2.0, one message per line).

Supports ``initialize``, ``ping``, ``tools/list`` and ``tools/call``. Notifications
(messages without an ``id``) get no reply. Output is compact JSON to keep the
transcript small.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .. import __version__
from .tools import call_tool, list_tools

PROTOCOL_VERSION = "2024-11-05"

__all__ = ["handle_message", "main"]


def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one decoded JSON-RPC message. Returns the reply, or None for notifications."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    method = msg["method"]
    req_id = msg.get("id")
    is_notification = "id" not in msg
    params = msg.get("params") or {}

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "finlab", "version": __version__},
        }
        return None if is_notification else _ok(req_id, result)
    if method == "notifications/initialized" or method.startswith("notifications/"):
        return None
    if method == "ping":
        return None if is_notification else _ok(req_id, {})
    if method == "tools/list":
        return None if is_notification else _ok(req_id, {"tools": list_tools()})
    if method == "tools/call":
        name = params.get("name")
        result, is_error = call_tool(str(name), params.get("arguments"))
        body = {
            "content": [{"type": "text", "text": json.dumps(result, separators=(",", ":"))}],
            "isError": is_error,
        }
        return None if is_notification else _ok(req_id, body)
    return None if is_notification else _err(req_id, -32601, f"method not found: {method}")


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    """Serve until stdin closes."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            reply: dict[str, Any] | None = _err(None, -32700, f"parse error: {exc.msg}")
        else:
            reply = handle_message(msg)
        if reply is not None:
            stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
