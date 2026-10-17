"""
hooks.py - Hook 系统：加载配置、匹配事件、执行 hook 命令
"""
import json
import os
import subprocess
from pathlib import Path

from loguru import logger

WORKDIR = Path.cwd()

HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SessionStart")
HOOK_TIMEOUT = 30


class HookManager:
    """加载 json 并在事件触发时执行匹配的 hook 命令。"""

    def __init__(self, config_path: Path = None):
        self.hooks = {event: [] for event in HOOK_EVENTS}
        config_path = config_path or (WORKDIR / "agent.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for event in HOOK_EVENTS:
                    self.hooks[event] = config.get("hooks", {}).get(event, [])
                logger.info("[hooks] 已加载 {}, {} 条 hook", config_path, sum(len(v) for v in self.hooks.values()))
            except Exception as e:
                logger.warning("[hooks] 配置解析失败: {}", e)

    def run_hooks(self, event: str, context: dict = None) -> dict:
        """
        执行某个事件的所有匹配 hook。
        返回: {"blocked": bool, "messages": list[str], "block_reason": str|None}
          - blocked: 有 hook 返回 exit code 1
          - messages: exit-code-2 hook 的 stderr 内容（注入到对话）
          - permission_override: hook stdout JSON 中的 permissionDecision
        """
        result = {"blocked": False, "messages": [], "block_reason": None}

        context = context or {}
        hooks = self.hooks.get(event, [])
        for hook_def in hooks:
            # matcher 保证hooks里面规定的工具名 和 context里面传进来的 一致
            matcher = hook_def.get("matcher")
            if matcher and matcher != "*":
                tool_name = context.get("tool_name", "")
                if matcher != tool_name:
                    continue

            command = hook_def.get("command", "")
            if not command:
                continue

            # 构建环境变量，注入 hook 上下文
            env = dict(os.environ)
            env["HOOK_EVENT"] = event
            env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
            env["HOOK_TOOL_INPUT"] = json.dumps(
                context.get("tool_input", {}), ensure_ascii=False)[:10000]
            if "tool_output" in context:
                env["HOOK_TOOL_OUTPUT"] = str(context["tool_output"])[:10000]

            try:
                r = subprocess.run(
                    command, shell=True, cwd=WORKDIR, env=env,
                    capture_output=True, text=True, timeout=HOOK_TIMEOUT,
                )
                if r.returncode == 0:
                    if r.stdout.strip():
                        logger.debug("[hook:{}] stdout: {}", event, r.stdout.strip()[:100])
                    # 尝试解析 structured stdout
                    try:
                        hook_output = json.loads(r.stdout)
                        if "updatedInput" in hook_output:
                            context["tool_input"] = hook_output["updatedInput"]
                        if "additionalContext" in hook_output:
                            result["messages"].append(hook_output["additionalContext"])
                        if "permissionDecision" in hook_output:
                            result["permission_override"] = hook_output["permissionDecision"]
                    except (json.JSONDecodeError, TypeError):
                        pass  # stdout 不是 JSON，正常情况
                elif r.returncode == 1:
                    result["blocked"] = True
                    reason = r.stderr.strip() or "Blocked by hook"
                    result["block_reason"] = reason
                    logger.info("[hook:{}] BLOCKED: {}", event, reason[:200])
                    break  # 一个 hook 阻止即停止
                elif r.returncode == 2:
                    msg = r.stderr.strip()
                    if msg:
                        result["messages"].append(msg)
                        logger.info("[hook:{}] INJECT: {}", event, msg[:200])
            except subprocess.TimeoutExpired:
                logger.warning("[hook:{}] 超时 ({}s)", event, HOOK_TIMEOUT)
            except Exception as e:
                logger.error("[hook:{}] 执行异常: {}", event, e)

        return result
