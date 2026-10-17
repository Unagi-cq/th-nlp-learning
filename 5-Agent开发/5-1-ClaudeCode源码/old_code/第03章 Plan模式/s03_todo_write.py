#!/usr/bin/env python3
"""
s03_todo_write.py - 会话计划与 TodoWrite

本章引入轻量级会话计划机制（非持久化任务图）。
模型可以重写当前计划、聚焦一个活跃步骤，
若多轮未刷新计划则会收到提醒。
"""
from config import client, MODEL
from loguru import logger
from tools import TOOLS, TOOL_HANDLERS, WORKDIR, TODO

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose."""


def _block_to_dict(block) -> dict | None:
    """将内容块统一转为 dict（兼容 SDK 对象和原始 dict）"""
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if not k.startswith("_")}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return None


def normalize_messages(messages: list) -> list:
    """在发送给 API 之前清理消息列表。

    三项工作：
    1. 将 SDK 对象转为 dict，剥离内部元数据字段
    2. 确保每个 tool_use 都有对应的 tool_result（缺失则插入占位符）
    3. 合并连续相同角色的消息（API 要求严格交替）
    """
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

    # 收集已有的 tool_result ID
    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    # 查找孤立的 tool_use 块并插入占位结果
    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") not in existing_results:
                logger.debug("插入占位 tool_result: tool_use_id={}", block["id"])
                cleaned.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"],
                     "content": "(cancelled)"}
                ]})

    # 合并连续相同角色的消息
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


def agent_loop(messages: list):
    """主循环：调用模型 -> 执行工具 -> 反馈结果 -> 继续，直到模型不再调用工具"""
    turn = 0
    while True:
        turn += 1
        logger.info("第 {} 轮开始", turn)
        logger.debug("模型原始输入：\n{}", messages)
        logger.debug("模型格式化输入：\n{}", normalize_messages(messages))
        response = client.messages.create(
            model=MODEL, 
            system=SYSTEM,
            messages=normalize_messages(messages),
            tools=TOOLS, 
            max_tokens=8000
        )

        logger.debug("模型响应: {}", response.to_json())
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            logger.info("第 {} 轮结束（模型无工具调用）", turn)
            return

        results = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                if handler:
                    logger.info("执行工具: {} | 参数: {}", block.name, block.input)
                    output = handler(**block.input)
                else:
                    logger.warning("未知工具: {}", block.name)
                    output = f"Unknown tool: {block.name}"
                logger.info(f"> {block.name}:")
                logger.info(output[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
                if block.name == "todo":
                    used_todo = True

        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                results.insert(0, {"type": "text", "text": reminder})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input(">> ")
        except (EOFError, KeyboardInterrupt):
            logger.info("用户中断，程序退出")
            break
        if query.strip().lower() in ("q", "exit", ""):
            logger.info("用户退出")
            break
        logger.info("用户输入: {}", query)
        
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    logger.info(block.text)
        print()