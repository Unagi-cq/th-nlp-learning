"""
compact.py - 上下文压缩工具，由模型主动调用来释放上下文空间。
实际压缩逻辑在 main.py 中拦截执行，此文件仅注册工具定义。
"""


def run_compact(focus: str = "") -> str:
    """占位 handler — main.py 拦截 compact 调用并执行实际压缩。"""
    return "[Compacted. Conversation history has been summarized.]"


TOOLS = [
    {
        "name": "compact",
        "description": "Summarize earlier conversation to free context space.",
        "handler": run_compact,
        "concurrency_safe": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to focus on in the summary, e.g. 'the bug fix in auth.py'.",
                },
            },
        },
    },
]
