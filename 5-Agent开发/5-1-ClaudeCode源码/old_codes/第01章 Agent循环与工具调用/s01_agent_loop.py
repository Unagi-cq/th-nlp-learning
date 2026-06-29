#!/usr/bin/env python3
"""
s01_agent_loop.py - 智能体循环

本文件演示最小可用的编码智能体模式：
    用户消息
      -> 模型回复
      -> 如果有 tool_use：执行工具
      -> 将 tool_result 写回消息列表
      -> 继续循环

循环状态保持显式声明，方便后续章节在此基础上扩展。
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

# ########## 模型与日志配置
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

logger.add("../app.log", rotation="5 MB", encoding="utf-8", level="DEBUG")

client = Anthropic(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
MODEL = os.getenv("MODEL_NAME")

# ########## 系统提示词与工具配置
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command in the current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


@dataclass
class LoopState:
    """循环状态：消息历史、轮次计数、继续原因"""
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def run_bash(command: str) -> str:
    """执行 shell 命令，拦截危险操作"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "rm"]
    if any(item in command for item in dangerous):
        logger.warning("危险命令已拦截: {}", command)
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.error("命令超时 (120s): {}", command)
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        logger.error("命令执行异常: {} | 错误: {}", command, e)
        return f"Error: {e}"
    output = (result.stdout + result.stderr).strip()
    logger.debug("命令完成: {} | 返回码: {} | 输出长度: {}", command, result.returncode, len(output))
    return output[:50000] if output else "(no output)"


def extract_text(content) -> str:
    """从响应内容中提取纯文本"""
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool_calls(response_content) -> list[dict]:
    """遍历响应中的工具调用并逐一执行"""
    results = []
    for block in response_content:
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        logger.info("执行工具调用: bash | 命令: {}", command)
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output)
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    """执行一轮对话：调用模型 -> 处理工具 -> 决定是否继续"""
    logger.info("第 {} 轮开始", state.turn_count)
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    logger.debug("模型输入：\n{}", state.messages)
    logger.debug("模型响应: \n{}", response.to_json())
    state.messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        state.transition_reason = None
        logger.info("第 {} 轮结束（模型无工具调用）", state.turn_count)
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True


def agent_loop(state: LoopState) -> None:
    """主循环：持续执行直到模型不再调用工具"""
    while run_one_turn(state):
        pass


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("s01 >>")
        except (EOFError, KeyboardInterrupt):
            logger.info("用户中断，程序退出")
            break
        if query.strip().lower() in ("q", "exit", ""):
            logger.info("用户退出")
            break
        logger.info("用户输入: {}", query)

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()