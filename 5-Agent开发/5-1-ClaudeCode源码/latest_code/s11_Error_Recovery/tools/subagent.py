from loguru import logger

from config import client, MODEL
from tools import maybe_spill

# 子 Agent 系统提示：强调只返回结论，不展开中间过程
SUB_SYSTEM = (
    "你是一个子 Agent，负责完成主 Agent 委派的子任务。"
    "你有文件读写、bash 等工具可用。"
    "完成后直接返回简洁的结论，不要展开中间推理过程。"
)


def _get_sub_handlers() -> dict:
    """返回子 Agent 可用的 handler 映射（不含 task，防止递归）。"""
    from tools.file import run_bash, run_read, run_write, run_edit, run_glob
    return {
        "bash": run_bash,
        "read_file": run_read,
        "write_file": run_write,
        "edit_file": run_edit,
        "glob": run_glob,
    }


def _get_sub_max_sizes() -> dict:
    from tools.file import TOOLS as file_tools
    return {tool["name"]: tool.get("maxResultSizeChars", 10_000) for tool in file_tools}


def _build_sub_tools() -> list[dict]:
    """从 file.py 的 TOOLS 中提取 API 格式的工具定义（去除 handler 等框架字段）。"""
    from tools.file import TOOLS as file_tools
    api_tools = []
    for t in file_tools:
        api_tools.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        })
    return api_tools


def _extract_text(content: list) -> str:
    """从模型返回的 content 列表中提取纯文本。"""
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def spawn_subagent(description: str) -> str:
    handlers = _get_sub_handlers()
    max_sizes = _get_sub_max_sizes()
    sub_tools = _build_sub_tools()
    messages = [{"role": "user", "content": description}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUB_SYSTEM,
            messages=messages,
            tools=sub_tools,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return _extract_text(response.content)

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 执行工具
            handler = handlers.get(block.name)
            params = dict(block.input) if block.input else {}
            try:
                output = handler(**params) if handler else f"未知工具: {block.name}"
                output = maybe_spill(block.name, output, max_sizes.get(block.name, 10_000))
            except Exception as e:
                logger.error(f"[子Agent] {block.name} 执行异常: {e}")
                output = f"工具执行异常: {e}"

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        messages.append({"role": "user", "content": results})

    # 超过最大轮次，返回最后一条消息的文本
    return _extract_text(messages[-1]["content"])


TOOLS = [
    {
        "name": "task",
        "description": "启动一个子 Agent 处理复杂的多步骤子任务。子 Agent 拥有文件读写、bash 等工具（无 task，防递归），仅返回最终结论。",
        "handler": spawn_subagent,
        "concurrency_safe": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "子任务的详细描述，子 Agent 根据此描述独立完成工作"},
            },
            "required": ["description"],
        },
    },
]
