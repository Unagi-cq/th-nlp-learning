"""
memory.py - 记忆查询工具。

模型可以先查看 system prompt 中的记忆索引，再按需调用此工具读取相关记忆正文。
"""

from memory import query_memory


TOOLS = [
    {
        "name": "query_memory",
        "description": (
            "查询长期记忆。可传 filenames 精确读取索引中的记忆文件；"
            "也可传 query 按关键词搜索相关记忆；不传参数时返回记忆索引。"
        ),
        "handler": query_memory,
        "concurrency_safe": True,
        "isReadOnly": True,
        "maxResultSizeChars": float("inf"),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于搜索记忆的关键词或自然语言问题。",
                },
                "filenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要精确读取的记忆文件名，例如 ['user-name.md']。",
                },
                "max_items": {
                    "type": "integer",
                    "description": "最多返回多少条记忆，默认 5，最大 10。",
                },
            },
        },
    },
]
