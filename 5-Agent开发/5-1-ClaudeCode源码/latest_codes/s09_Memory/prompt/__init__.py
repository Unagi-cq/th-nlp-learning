from datetime import datetime
from pathlib import Path

from memory import memory_store
from system.paths import PROJECT_ROOT
from tools.skills import skill_registry

_TEMPLATE = (Path(__file__).parent / "system.txt").read_text(encoding="utf-8")


def load_system_prompt() -> str:
    index = memory_store.read_index()
    memories_section = (
        "\n\n可用记忆索引:\n"
        f"{index}\n"
        "如需使用某条记忆，请调用 query_memory 工具读取相关记忆正文；不要假设索引已包含完整内容。"
    ) if index else ""

    return _TEMPLATE.format(
        pwd=PROJECT_ROOT,
        skills=skill_registry.describe_available(),
        current_date_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        memory=memories_section,
    )
