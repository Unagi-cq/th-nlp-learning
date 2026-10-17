#!/usr/bin/env python3
"""
s07_permission.py - 权限系统
"""
from config import client, MODEL
from loguru import logger
from tools import TOOLS, TOOL_HANDLERS, TODO, run_subagent
from tools import (
    CompactState,
    CONTEXT_LIMIT,
    micro_compact,
    estimate_context_size,
    compact_history,
    persist_large_output,
    track_recent_file,
    PermissionManager,
    MODES,
)
import display
from prompt import load_system_prompt

from utils import normalize_messages


def agent_loop(messages: list, state: CompactState, perms: PermissionManager):
    system = load_system_prompt()
    # 记录本轮开始位置，micro_compact 只处理本轮新增的工具结果
    cursor = len(messages)
    turn = 0
    while True:
        turn += 1
        display.turn_header(turn)

        # 微压缩：只截断本轮新增的旧工具结果，不动历史对话
        if len(messages) > cursor:
            before = messages[:cursor]
            after = micro_compact(messages[cursor:])
            messages[:] = before + after

        # 自动压缩：上下文超限时触发
        if estimate_context_size(messages) > CONTEXT_LIMIT:
            logger.info("[auto compact] 上下文超限，触发自动压缩")
            display.tool("auto-compact", {})
            messages[:] = compact_history(messages, state)
            cursor = 0

        response = client.messages.create(
            model=MODEL,
            system=system,
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
            if block.type == "thinking":
                display.thinking(block.thinking)
            elif block.type == "text":
                display.agent_text(block.text)
            elif block.type == "tool_use":
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
                    if not handler:
                        logger.warning("未知工具: {}", block.name)
                        output = f"Unknown tool: {block.name}"
                    else:
                        decision = perms.check(block.name, params)
                        if decision["behavior"] == "deny":
                            output = f"Permission denied: {decision['reason']}"
                            display.permission_denied(block.name, decision["reason"])
                        elif decision["behavior"] == "ask":
                            if perms.ask_user(block.name, params):
                                output = handler(**params)
                                output = persist_large_output(block.id, output)
                            else:
                                output = f"Permission denied by user for {block.name}"
                                display.permission_user_denied(block.name)
                        else:
                            output = handler(**params)
                            output = persist_large_output(block.id, output)
                    if block.name == "read_file":
                        track_recent_file(state, params.get("path", ""))
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

        if manual_compact:
            logger.info("[manual compact] 手动触发压缩")
            messages[:] = compact_history(messages, state, focus=compact_focus)
            cursor = 0

if __name__ == "__main__":
    from tools import WORKDIR, SKILL_REGISTRY
    display.banner(MODEL, str(WORKDIR), SKILL_REGISTRY.count, SKILL_REGISTRY.names)

    # 启动时选择权限模式
    print(f"Permission modes: {', '.join(MODES)}")
    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in MODES:
        mode_input = "default"
    perms = PermissionManager(mode=mode_input)
    print(f"[Permission mode: {mode_input}]")

    history = []
    compact_state = CompactState()
    while True:
        try:
            query = display.user_prompt()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() in ("q", "exit"):
            break
        logger.info("用户输入: {}", query)

        # /mode 运行时切换权限模式
        if query.startswith("/mode"):
            parts = query.split()
            if len(parts) == 2 and parts[1] in MODES:
                perms.mode = parts[1]
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"Usage: /mode <{'|'.join(MODES)}>")
            continue

        # /rules 查看当前规则
        if query.strip() == "/rules":
            for i, rule in enumerate(perms.rules):
                print(f"  {i}: {rule}")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state, perms)
