"""
tools.py - 工具定义、实现与分发表
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

WORKDIR = Path.cwd()

# -- 并发安全分类 --
# 只读工具可以安全并行执行；写入工具必须串行
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}
PLAN_REMINDER_INTERVAL = 3  # 连续多少轮未更新计划后触发提醒


def safe_path(p: str) -> Path:
    """将相对路径解析为绝对路径，并确保不会逃逸出工作目录"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        logger.warning("路径逃逸工作目录: {}", p)
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，拦截危险操作"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        logger.warning("[bash] 危险命令已拦截: {}", command)
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        logger.debug("[bash] {} → 返回码={} 输出={}字节", command[:60], r.returncode, len(out))
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        logger.error("[bash] 超时 (120s): {}", command[:60])
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容，可选限制行数"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        total = len(lines)
        if limit and limit < total:
            lines = lines[:limit] + [f"... ({total - limit} more lines)"]
        logger.debug("[read] {} ({} 行)", path, total)
        return "\n".join(lines)[:50000]
    except Exception as e:
        logger.error("[read] {} | {}", path, e)
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入文件，自动创建父目录"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        logger.info("[write] {} ({} 字节)", path, len(content))
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        logger.error("[write] {} | {}", path, e)
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换文本（仅替换第一处匹配）"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            logger.warning("[edit] 未找到目标文本: {}", path)
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        logger.info("[edit] {}", path)
        return f"Edited {path}"
    except Exception as e:
        logger.error("[edit] {} | {}", path, e)
        return f"Error: {e}"

# ===========================================
# 计划管理（TodoManager）
# ===========================================

@dataclass
class PlanItem:
    """单条计划项"""
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    """计划整体状态"""
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    """会话计划管理器：维护计划列表、渲染状态、检测超时未更新"""

    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        """重写整个计划列表，校验约束后更新状态"""
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")
        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()
            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        self.state.items = normalized
        self.state.rounds_since_update = 0
        logger.info("[todo] 计划已更新 ({} 项)", len(normalized))
        return self.render()

    def note_round_without_update(self) -> None:
        """记录一轮未更新计划"""
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        """若超过阈值轮次未更新计划，返回提醒文本"""
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        logger.debug("[todo] 提醒触发 (已 {} 轮未更新)", self.state.rounds_since_update)
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        """将当前计划渲染为可读文本"""
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# ===========================================
# 工具定义
# ===========================================

# 工具分发表：{工具名: 处理函数}
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
}

# 工具 Schema 定义（供 API 调用）
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short description of the task"
                }
            },
            "required": ["prompt"]
        }
    }
]
