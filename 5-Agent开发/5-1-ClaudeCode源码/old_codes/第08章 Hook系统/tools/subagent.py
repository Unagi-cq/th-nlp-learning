"""
subagent.py - 子代理：独立上下文执行任务
"""
from config import client, MODEL
from loguru import logger
from .file import TOOLS as _F_TOOLS, TOOL_HANDLERS as _F_HANDLERS, WORKDIR
from .plan import TOOLS as _P_TOOLS, TOOL_HANDLERS as _P_HANDLERS
from .skills import TOOLS as _S_TOOLS, TOOL_HANDLERS as _S_HANDLERS
from .compact import TOOLS as _C_TOOLS, TOOL_HANDLERS as _C_HANDLERS
import display

ALL_TOOLS = _F_TOOLS + _P_TOOLS + _S_TOOLS + _C_TOOLS
ALL_HANDLERS = {**_F_HANDLERS, **_P_HANDLERS, **_S_HANDLERS, **_C_HANDLERS}

# 子代理不需要 plan task compact
SUBAGENT_TOOLS = [t for t in ALL_TOOLS if t["name"] not in {"todo", "task", "compact"}]
SUBAGENT_HANDLERS = {k: v for k, v in ALL_HANDLERS.items() if k not in {"todo", "compact"}}

SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


def run_subagent(desc: str, prompt: str) -> str:
    display.task_delegate(desc, prompt)

    sub_messages = [{"role": "user", "content": prompt}]
    for turn in range(1, 31):
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=SUBAGENT_TOOLS,
            max_tokens=8000
        )
        sub_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            summary = "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"
            logger.info("[子代理] 完成 ({} 轮) → {}", turn, summary[:120])
            display.subagent_done(turn, summary)
            return summary

        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = SUBAGENT_HANDLERS.get(block.name)
                params = dict(block.input) if block.input else {}
                param_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in params.items())
                logger.debug("[子代理] {}({})", block.name, param_str)
                display.subagent_log(f"[{block.name}] {param_str}")
                output = handler(**params) if handler else f"Unknown tool: {block.name}"
                if output:
                    logger.debug("[子代理]   → {}", output[:100])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})

    return "(subagent hit turn limit)"


TOOL_HANDLERS = {
    "task": lambda **kw: run_subagent(kw.get("description", "subtask"), kw["prompt"]),
}

TOOLS = [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short description of the task"
                }
            },
            "required": ["prompt"]
        }
    },
]
