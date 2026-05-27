from config import client, MODEL
from loguru import logger
from tools import TOOLS, TOOL_HANDLERS, WORKDIR
import display

# 子代理不需要 plan task compact
SUBAGENT_TOOLS = [t for t in TOOLS if t["name"] not in {"todo", "task", "compact"}]

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
                handler = TOOL_HANDLERS.get(block.name)
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
