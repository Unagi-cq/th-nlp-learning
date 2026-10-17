import json
from datetime import datetime

from loguru import logger

from memory import memory_store
from system.paths import PROJECT_ROOT
from tools.skills import skill_registry

PROMPT_SECTIONS = {
    "identity": (
        "你是位于当前工作区的编程助手。使用工具和技能来解决问题。"
        "直接执行，不要解释。"
    ),
    "tools": (
        "可用工具由运行时工具注册表提供。需要读取、修改、搜索、执行、查询记忆或加载技能时，"
        "优先调用对应工具。"
    ),
    "workspace": "当前工作区: {workspace}",
    "skills": "可用技能列表:\n{skills}",
    "memory": (
        "可用记忆索引:\n{memories}\n"
        "如需使用某条记忆，请调用 query_memory 工具读取相关记忆正文；"
        "不要假设索引已包含完整内容。"
    ),
    "runtime": "当前日期: {current_date}",
}

_last_context_key = ""
_last_prompt = ""


def _enabled_tool_names() -> list[str]:
    from tools import tool_registry

    return sorted(tool["name"] for tool in tool_registry.to_api_list())


def build_prompt_context(messages: list | None = None) -> dict:
    """读取真实运行态，构造 prompt context。"""
    memories = memory_store.read_index()
    skills = skill_registry.describe_available()

    return {
        "workspace": str(PROJECT_ROOT),
        "enabled_tools": _enabled_tool_names(),
        "skills": skills if skills != "(no skills available)" else "",
        "memories": memories,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }


def update_context(context: dict | None, messages: list | None = None) -> dict:
    """根据真实状态刷新 context；不根据用户消息关键词猜测 section。"""
    next_context = build_prompt_context(messages)
    if context:
        next_context.update({
            key: value
            for key, value in context.items()
            if key not in next_context
        })
    return next_context


def assemble_system_prompt(context: dict) -> str:
    """按需拼接 system prompt sections。"""
    section_names = ["identity", "tools", "workspace", "runtime"]
    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["tools"],
        PROMPT_SECTIONS["workspace"].format(workspace=context["workspace"]),
        PROMPT_SECTIONS["runtime"].format(current_date=context["current_date"]),
    ]

    enabled_tools = context.get("enabled_tools") or []
    if enabled_tools:
        sections.append("已启用工具:\n" + "\n".join(f"- {name}" for name in enabled_tools))
        section_names.append("enabled_tools")

    skills = context.get("skills", "")
    if skills:
        sections.append(PROMPT_SECTIONS["skills"].format(skills=skills))
        section_names.append("skills")

    memories = context.get("memories", "")
    if memories:
        sections.append(PROMPT_SECTIONS["memory"].format(memories=memories))
        section_names.append("memory")

    logger.debug("[prompt assembled] sections: {}", ", ".join(section_names))
    return "\n\n".join(sections)


def get_system_prompt(context: dict) -> str:
    """缓存 system prompt，context 未变化时直接复用。"""
    global _last_context_key, _last_prompt

    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        logger.debug("[prompt cache hit]")
        return _last_prompt

    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


def load_system_prompt(messages: list | None = None, context: dict | None = None) -> str:
    prompt_context = update_context(context, messages)
    return get_system_prompt(prompt_context)
