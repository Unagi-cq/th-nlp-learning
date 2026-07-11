import importlib
import pkgutil
from collections import defaultdict


class Hook:
    """Hook 定义：事件名 + 回调函数，类似 Tool 的数据类。"""

    def __init__(self, event: str, callback, name: str = ""):
        self.event = event
        self.callback = callback
        self.name = name or callback.__name__

    def __call__(self, *args):
        return self.callback(*args)


class HookRegistry:
    """Hook 注册中心：管理所有 hook 的注册与触发，类似 ToolRegistry。"""

    def __init__(self):
        self._hooks: dict[str, list[Hook]] = defaultdict(list)

    def register(self, hook: Hook):
        self._hooks[hook.event].append(hook)

    def trigger(self, event: str, *args):
        """触发某事件的所有 hook。返回第一个非 None 值，否则返回 None。"""
        for hook in self._hooks[event]:
            result = hook(*args)
            if result is not None:
                return result
        return None

    def list(self, event: str = None) -> list[Hook]:
        """列出已注册的 hook。不传 event 则返回全部。"""
        if event:
            return self._hooks.get(event, [])
        return [h for hooks in self._hooks.values() for h in hooks]


hook_registry = HookRegistry()


def _module_to_event(module_name: str) -> str:
    """user_prompt_submit → UserPromptSubmit"""
    return "".join(part.capitalize() for part in module_name.split("_"))


# 自动扫描所有子模块，从文件名推导 event，注册 HOOKS 中的回调
for mod_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    mod = importlib.import_module(mod_info.name)
    if hasattr(mod, "HOOKS"):
        event = _module_to_event(mod_info.name.split(".")[-1])
        for callback in mod.HOOKS:
            hook_registry.register(Hook(event, callback))
