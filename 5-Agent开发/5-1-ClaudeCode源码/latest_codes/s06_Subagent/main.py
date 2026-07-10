"""
最小 Agent Loop 实现
一个 while 循环 + bash 工具 + 并发安全分区 + batch 内并发执行
"""
from loguru import logger

from config import client, MODEL
from hooks import hook_registry
from permission import permission_manager
from prompt import load_system_prompt
from state import State
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


def agent_loop(state: State):
    state.round_num = 0

    while True:
        state.round_num += 1
        logger.info(f"  [轮次 {state.round_num}] 调用模型...")
        response = client.messages.create(
            model=MODEL,
            system=load_system_prompt(),
            messages=state.messages,
            tools=tool_registry.to_api_list(),
            max_tokens=8000
        )

        state.messages.append({"role": "assistant", "content": response.content})

        # ########### 循环退出前的处理
        if response.stop_reason != "tool_use":
            force = hook_registry.trigger("Stop", state)  # hook 退出之前
            if force:
                state.messages.append({"role": "user", "content": force})

            for block in response.content:
                if getattr(block, "type", None) == "text":
                    logger.opt(raw=True).info(block.text + "\n")
            return

        # ########### 收集工具调用请求
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_count = len(tool_blocks)
        logger.info(f"  [轮次 {state.round_num}] 模型请求 {tool_count} 个工具调用")

        # ########### 权限检查 在主线程中逐個检查，支持用户交互
        approved_blocks = []
        denied_results = []
        for block in tool_blocks:
            if permission_manager.check_permission(block):
                approved_blocks.append(block)
            else:
                logger.info(f"  [权限] {block.name} 被拒绝")
                denied_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Permission denied.",
                })

        results = denied_results[:]

        # ########### 执行工具调用
        if approved_blocks:
            batches = partition_tool_calls(approved_blocks)
            logger.info(f"  [工具调用分区] → {len(batches)} 个 batch: {[[b.name for b in batch] for batch in batches]}")

            for i, batch in enumerate(batches):
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


if __name__ == "__main__":
    logger.info(f"当前权限模式: {permission_manager.mode.value}")

    queries = ["用 子agent 看看桌面目录下有哪些txt文件 然后读取第2个txt 为我写读后感"]
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

        hook_registry.trigger("UserPromptSubmit", query)  # hook 进入 LLM 之前

        state.messages.append({"role": "user", "content": query})
        agent_loop(state)
        print()
