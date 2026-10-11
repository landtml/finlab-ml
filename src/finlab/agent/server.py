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


def _reject_constant(name: str) -> None:
    # json.loads accepts NaN and Infinity by default; they are not JSON and would
    # be echoed back as invalid output.
    raise ValueError(f"non-standard constant {name}")


def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one decoded JSON-RPC message. Returns the reply, or None for notifications."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    method = msg["method"]
    req_id = msg.get("id")
    is_notification = "id" not in msg
    params = msg.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None if is_notification else _err(req_id, -32602, "invalid params: expected an object")

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
        if not isinstance(name, str):
            return None if is_notification else _err(req_id, -32602, "invalid params: missing tool name")
        result, is_error = call_tool(name, params.get("arguments"))
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
            msg = json.loads(line, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            reply: dict[str, Any] | None = _err(None, -32700, f"parse error: {exc.msg}")
        except RecursionError:
            # Nesting deeper than the decoder allows; the line is not valid JSON for us.
            reply = _err(None, -32700, "parse error: nesting too deep")
        except ValueError as exc:
            # e.g. an integer literal over the decoder's digit limit.
            reply = _err(None, -32700, f"parse error: {exc}")
        else:
            try:
                reply = handle_message(msg)
            except Exception as exc:  # one bad message must not stop the server
                is_request = isinstance(msg, dict) and "id" in msg
                reply = _err(msg.get("id"), -32603, f"internal error: {type(exc).__name__}") if is_request else None
        if reply is not None:
            stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
