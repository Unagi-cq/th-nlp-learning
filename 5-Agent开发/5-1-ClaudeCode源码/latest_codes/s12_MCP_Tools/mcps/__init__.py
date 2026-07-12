import atexit
import json
import re
import subprocess
import threading
from pathlib import Path

from loguru import logger

MCP_CONFIG = Path(__file__).parent / ".mcp.json"
_BAD_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_]")
_clients = []


def _safe_name(name: str) -> str:
    return _BAD_NAME_CHARS.sub("_", name)


def _tool_name(server: str, tool: str) -> str:
    return f"mcp__{_safe_name(server)}__{_safe_name(tool)}"


def _read_config() -> dict:
    if not MCP_CONFIG.exists():
        return {}
    config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
    return config.get("servers", config)


def _command(cfg: dict) -> list[str]:
    command = cfg.get("command")
    if isinstance(command, list):
        return [str(part) for part in command]
    args = cfg.get("args", [])
    return [str(command), *[str(arg) for arg in args]]


class MCPClient:
    def __init__(self, name: str, command: list[str]):
        self.name = name
        self.command = command
        self.proc: subprocess.Popen | None = None
        self.next_id = 1
        self.lock = threading.Lock()

    def start(self) -> list[dict]:
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mini-agent", "version": "0.1.0"},
        })
        self.notify("notifications/initialized", {})
        return self.request("tools/list", {}).get("tools", [])

    def call(self, tool: str, arguments: dict) -> str:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        blocks = result.get("content", [])
        texts = [
            block.get("text", "") if block.get("type") == "text" else json.dumps(block, ensure_ascii=False)
            for block in blocks
        ]
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)

    def request(self, method: str, params: dict) -> dict:
        with self.lock:
            request_id = self.next_id
            self.next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            while True:
                message = self._recv()
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def notify(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _send(self, message: dict):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"MCP server '{self.name}' exited. {stderr}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("[mcp:{}] non-json stdout: {}", self.name, line.strip())


def _register_tools(server_name: str, client: MCPClient, tools: list[dict]) -> list[str]:
    from tools import Tool, tool_registry

    registered = []
    for tool in tools:
        original_name = tool["name"]
        registered_name = _tool_name(server_name, original_name)
        raw = {
            "name": registered_name,
            "description": f"[MCP:{server_name}] {tool.get('description', '')}".strip(),
            "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
            "handler": lambda _tool=original_name, _client=client, **kwargs: _client.call(_tool, kwargs),
            "concurrency_safe": True,
            "isReadOnly": True,
        }
        tool_registry.register(Tool(raw))
        registered.append(registered_name)
    return registered


def load_enabled_mcp_servers() -> list[str]:
    registered = []
    for name, cfg in _read_config().items():
        if not cfg.get("enabled", True):
            continue
        try:
            client = MCPClient(name, _command(cfg))
            tools = client.start()
            _clients.append(client)
            registered.extend(_register_tools(name, client, tools))
            logger.info("[mcp] connected {}: {} tools", name, len(tools))
        except Exception as exc:
            logger.warning("[mcp] failed to connect {}: {}", name, exc)
    return registered


atexit.register(lambda: [client.close() for client in _clients])
