"""
最小 Agent Loop 实现
一个 while 循环 + bash 工具 + 并发安全分区 + batch 内并发执行
"""
from loguru import logger

from config import client, MODEL
from permission import permission_manager, PermissionMode
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

        if response.stop_reason != "tool_use":
            logger.info(f"  [轮次 {state.round_num}] 模型返回文本回复，对话结束")
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    logger.opt(raw=True).info(block.text + "\n")
            return

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

        # ########### 工具调用
        if approved_blocks:
            batches = partition_tool_calls(approved_blocks)

            logger.info(f"  [工具调用分区] → {len(batches)} 个 batch: {[[b.name for b in batch] for batch in batches]}")

            for i, batch in enumerate(batches):
                results.extend(execute_batch(batch))

        state.messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # 选择权限模式
    print("选择权限模式:")
    print("  1 - 自动拒绝（所有工具调用直接拒绝）")
    print("  2 - 学习模式（记住用户选择，后续自动应用）")
    choice = input("请输入 [1/2，默认 2]: ").strip()
    mode_map = {"1": PermissionMode.AUTO_DENY, "2": PermissionMode.LEARNING}
    permission_manager.set_mode(mode_map.get(choice, PermissionMode.LEARNING))
    logger.info(f"当前权限模式: {permission_manager.mode.value}")

    queries = ["你好", "你有哪些工具", "看看目录目录下有哪些文件", "找我桌面下的所有pdf文件，然后打印最大的pdf的全部文件内容"]
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

        state.messages.append({"role": "user", "content": query})
        agent_loop(state)
        print()
