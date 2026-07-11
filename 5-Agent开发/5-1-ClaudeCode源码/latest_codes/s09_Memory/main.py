"""
最小 Agent Loop 实现
一个 while 循环 + bash 工具 + 并发安全分区 + batch 内并发执行
"""

from loguru import logger

from config import client, MODEL
from hooks import hook_registry
from memory import schedule_memory_maintenance
from prompt import load_system_prompt
from state import State
from system.compact import snip_compact, estimate_size, CONTEXT_LIMIT, \
    compact_history, reactive_compact
from system.permission import permission_manager
from tools import tool_registry, partition_tool_calls, execute_batch

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

MAX_REACTIVE_RETRIES = 1  # LLM重试次数
MAX_AGENT_ROUNDS = 30


def agent_loop(state: State):
    state.round_num = 0

    while state.round_num < MAX_AGENT_ROUNDS:
        state.round_num += 1
        # hook 模型调用之前
        hook_registry.trigger("PreModelCall", state, MODEL)

        # ########### 上下文压缩
        # L1: 消息数裁剪
        state.messages[:] = snip_compact(state.messages)

        # L2: 大模型压缩 tokens数量仍然超过阈值 那么调用LLM生成摘要
        if estimate_size(state.messages) > CONTEXT_LIMIT:
            state.messages[:] = compact_history(state.messages)

        # ########### 模型调用 含重试机制与压缩机制
        try:
            response = client.messages.create(
                model=MODEL,
                system=load_system_prompt(),
                messages=state.messages,
                tools=tool_registry.to_api_list(),
                max_tokens=8000
            )
            state.reactive_retries = 0
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and state.reactive_retries < MAX_REACTIVE_RETRIES:
                state.messages[:] = reactive_compact(state.messages)
                state.reactive_retries += 1
                continue
            raise
        state.messages.append({"role": "assistant", "content": response.content})

        # ########### 循环退出前的处理
        if response.stop_reason != "tool_use":
            # hook 退出之前
            force = hook_registry.trigger("Stop", state)
            if force:
                state.messages.append({"role": "user", "content": force})
                continue

            for block in response.content:
                if getattr(block, "type", None) == "text":
                    logger.opt(raw=True).info(block.text + "\n")

            # 异步执行 记忆提取
            schedule_memory_maintenance(state.messages)
            return

        # ########### 收集工具调用请求
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_count = len(tool_blocks)
        logger.info(f"[轮次 {state.round_num}] 模型请求 {tool_count} 个工具调用")

        # ########### compact 工具：模型主动压缩上下文，立即结束当前轮次
        compact_block = next((b for b in tool_blocks if b.name == "compact"), None)
        if compact_block:
            focus = dict(compact_block.input).get("focus", "") if compact_block.input else ""
            state.messages[:] = compact_history(state.messages, focus)
            continue

        # ########### 权限检查 在主线程中逐個检查，支持用户交互
        approved_blocks = []
        denied_results = []
        for block in tool_blocks:
            if permission_manager.check_permission(block):
                approved_blocks.append(block)
            else:
                logger.info(f"[权限] {block.name} 被拒绝")
                denied_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Permission denied.",
                })

        results = denied_results[:]

        # ########### 执行工具调用
        if approved_blocks:
            batches = partition_tool_calls(approved_blocks)
            logger.info(f"[工具调用分区] → {len(batches)} 个 batch: {[[b.name for b in batch] for batch in batches]}")

            for batch in batches:
                results.extend(execute_batch(batch))

        state.messages.append({"role": "user", "content": results})

        # ########### todowrite 提醒：本轮回调用了 todo_write 则重置计数，否则累加
        if any(b.name == "todo_write" for b in approved_blocks):
            state.rounds_since_todo = 0
        else:
            state.rounds_since_todo += 1

        if state.rounds_since_todo >= 3 and state.messages:
            state.messages.append({
                "role": "user",
                "content": "<reminder>请更新你的 todo 任务列表。</reminder>",
            })
            state.rounds_since_todo = 0

    logger.warning("[agent] 超过最大轮次 {}，强制结束当前对话", MAX_AGENT_ROUNDS)


if __name__ == "__main__":
    logger.info(f"当前权限模式: {permission_manager.mode.value}")

    queries = ["你有哪些技能", "读取我桌面文件夹下的文件 生成我的用户画像"]
    state = State()
    i = 0
    while True and i < len(queries):
        try:
            query = queries[i]
            i = i + 1
            state.conv_num += 1
            logger.info(f"[对话 {state.conv_num}] {query}")
        except (EOFError, KeyboardInterrupt):
            break

        # hook 进入 LLM 之前
        hook_registry.trigger("UserPromptSubmit", query)

        state.messages.append({"role": "user", "content": query})
        agent_loop(state)
        print()
