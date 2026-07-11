from loguru import logger

CURRENT_TODOS: list[dict] = []


def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    CURRENT_TODOS = [todo for todo in todos if isinstance(todo, dict)]

    lines = ["\n## Current Tasks"]
    for t in CURRENT_TODOS:
        status = t.get("status", "pending")
        icon = {"pending": " ", "in_progress": "▸", "completed": "✓"}.get(status, "?")
        lines.append(f"[{icon}] {t.get('content', '')}")
    logger.info("\n".join(lines))
    return f"已更新 {len(CURRENT_TODOS)} 个任务"


TOOLS = [
    {
        "name": "todo_write",
        "description": "创建和管理任务列表，用于规划和追踪 Agent Loop 中的多步骤任务进度。",
        "handler": run_todo_write,
        "concurrency_safe": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "任务内容描述"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "任务状态：pending=待办, in_progress=进行中, completed=已完成",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
]
