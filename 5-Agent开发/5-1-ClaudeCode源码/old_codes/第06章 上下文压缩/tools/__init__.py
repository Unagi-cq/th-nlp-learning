import importlib
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

# 自动扫描所有子模块，聚合 TOOLS 和 TOOL_HANDLERS
TOOLS = []
TOOL_HANDLERS = {}

for mod_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    mod = importlib.import_module(mod_info.name)
    if hasattr(mod, "TOOLS"):
        TOOLS.extend(mod.TOOLS)
    if hasattr(mod, "TOOL_HANDLERS"):
        TOOL_HANDLERS.update(mod.TOOL_HANDLERS)
