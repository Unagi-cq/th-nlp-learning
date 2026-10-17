from loguru import logger

from state import State


def summary_hook(state: State) -> str | None:
    logger.opt(raw=True).info(f"\033[90m[HOOK] Stop: [轮次 {state.round_num}] 模型返回文本回复，对话结束\033[0m\n")
    tool_count = sum(1 for m in state.messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    logger.opt(raw=True).info(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m\n")
    return None


HOOKS = [summary_hook]
