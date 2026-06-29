import importlib
import json
import pkgutil

from .file import WORKDIR
from .plan import TODO
from .skills import SKILL_REGISTRY
from .compact import (
    CompactState,
    CONTEXT_LIMIT,
    micro_compact,
    estimate_context_size,
    compact_history,
    persist_large_output,
    track_recent_file,
)
from .subagent import run_subagent

# 加载 agent.json 配置
_config_path = WORKDIR / "agent.json"
_agent_config = {}
if _config_path.exists():
    try:
        _agent_config = json.loads(_config_path.read_text())
    except Exception:
        pass

COMPACT_ENABLED = _agent_config.get("compact_enable", True)
PLAN_ENABLED = _agent_config.get("plan_enable", True)
SUBAGENT_ENABLED = _agent_config.get("subagent_enable", True)

# 自动扫描所有子模块，聚合 TOOLS 和 TOOL_HANDLERS
TOOLS = []
TOOL_HANDLERS = {}

for mod_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    mod = importlib.import_module(mod_info.name)
    if hasattr(mod, "TOOLS"):
        TOOLS.extend(mod.TOOLS)
    if hasattr(mod, "TOOL_HANDLERS"):
        TOOL_HANDLERS.update(mod.TOOL_HANDLERS)

# 按开关过滤
if not COMPACT_ENABLED:
    TOOLS = [t for t in TOOLS if t["name"] != "compact"]
    TOOL_HANDLERS.pop("compact", None)
if not PLAN_ENABLED:
    TOOLS = [t for t in TOOLS if t["name"] != "todo"]
    TOOL_HANDLERS.pop("todo", None)
if not SUBAGENT_ENABLED:
    TOOLS = [t for t in TOOLS if t["name"] != "task"]
    TOOL_HANDLERS.pop("task", None)
