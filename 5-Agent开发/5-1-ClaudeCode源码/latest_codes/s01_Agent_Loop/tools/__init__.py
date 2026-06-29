import importlib
import pkgutil

# 自动扫描所有子模块，聚合 TOOLS 和 TOOL_HANDLERS
TOOLS = []
TOOL_HANDLERS = {}

for mod_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    mod = importlib.import_module(mod_info.name)
    if hasattr(mod, "TOOLS"):
        TOOLS.extend(mod.TOOLS)
    if hasattr(mod, "TOOL_HANDLERS"):
        TOOL_HANDLERS.update(mod.TOOL_HANDLERS)
