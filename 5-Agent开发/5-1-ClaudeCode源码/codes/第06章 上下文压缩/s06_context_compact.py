#!/usr/bin/env python3
"""
s06_context_compact.py - 上下文压缩
"""
from config import client, MODEL
from loguru import logger
from tools import TOOLS, TOOL_HANDLERS, WORKDIR, TODO
from subagent import run_subagent
from skills import SKILL_REGISTRY
import display
import compact

SYSTEM = f"""You are a coding agent at {WORKDIR}.

- Use tools to solve tasks. Act, don't explain.
- Use the todo tool for multi-step work. Keep exactly one step in_progress when a task has multiple steps. Refresh the plan as work advances. Prefer tools over prose.
- Use the task tool to delegate exploration or subtasks.
- Keep working step by step, and use compact if the conversation gets too long.
- Use load_skill when a task needs specialized instructions before you act.
Skills available:
{SKILL_REGISTRY.describe_available()}
"""


def _block_to_dict(block) -> dict | None:
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if not k.startswith("_")}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return None


def normalize_messages(messages: list) -> list:
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            blocks = []
            for block in msg["content"]:
                converted = _block_to_dict(block)
                if converted:
                    blocks.append(converted)
            clean["content"] = blocks
        else:
            clean["content"] = msg.get("content", "")
        cleaned.append(clean)

    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") not in existing_results:
                cleaned.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"], "content": "(cancelled)"}
                ]})

    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": str(prev["content"])}]
            curr_c = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged


def agent_loop(messages: list, state: compact.CompactState):
    turn = 0
    while True:
        turn += 1
        display.turn_header(turn)

        # 微压缩：截断旧的工具结果
        messages[:] = compact.micro_compact(messages)

        # 自动压缩：上下文超限时触发
        if compact.estimate_context_size(messages) > compact.CONTEXT_LIMIT:
            logger.info("[auto compact] 上下文超限，触发自动压缩")
            display.tool("auto-compact", {})
            messages[:] = compact.compact_history(messages, state)

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=normalize_messages(messages),
            tools=TOOLS,
            max_tokens=8000
        )
        logger.debug("turn {} response: stop_reason={}", turn, response.stop_reason)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text.strip():
                display.agent_text(text)
            display.agent_stop(response.stop_reason)
            return

        results = []
        used_todo = False
        manual_compact = False
        compact_focus = None
        for block in response.content:
            if block.type == "tool_use":
                params = dict(block.input) if block.input else {}

                if block.name == "task":
                    desc = params.get("description", "subtask")
                    prompt = params.get("prompt", "")
                    output = run_subagent(desc, prompt)

                elif block.name == "todo":
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**params)
                    display.plan(output)
                    used_todo = True

                elif block.name == "compact":
                    manual_compact = True
                    compact_focus = params.get("focus")
                    output = "Compacting conversation..."
                    display.tool("compact", params)

                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    if handler:
                        output = handler(**params)
                    else:
                        logger.warning("未知工具: {}", block.name)
                        output = f"Unknown tool: {block.name}"
                    # 大输出持久化到磁盘
                    output = compact.persist_large_output(block.id, output)
                    # 记录最近读取的文件
                    if block.name == "read_file":
                        compact.track_recent_file(state, params.get("path", ""))
                    display.tool(block.name, params)
                    display.tool_output(output)

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                results.insert(0, {"type": "text", "text": reminder})
        messages.append({"role": "user", "content": results})

        # 手动压缩：本轮工具调用完成后执行
        if manual_compact:
            logger.info("[manual compact] 手动触发压缩")
            messages[:] = compact.compact_history(messages, state, focus=compact_focus)

if __name__ == "__main__":
    display.banner(MODEL, str(WORKDIR), SKILL_REGISTRY.count, SKILL_REGISTRY.names)

    history = []
    compact_state = compact.CompactState()
    while True:
        query = display.user_prompt()
        logger.info("用户输入: {}", query)

        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)
