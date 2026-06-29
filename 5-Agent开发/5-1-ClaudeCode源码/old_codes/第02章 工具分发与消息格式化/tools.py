"""
tools.py - 工具定义、实现与分发表
"""
import subprocess
from pathlib import Path

from loguru import logger

WORKDIR = Path.cwd()

# -- 并发安全分类 --
# 只读工具可以安全并行执行；写入工具必须串行
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}


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
        logger.warning("危险命令已拦截: {}", command)
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        logger.debug("bash 执行完成: {} | 返回码: {} | 输出长度: {}", command, r.returncode, len(out))
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        logger.error("命令超时 (120s): {}", command)
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容，可选限制行数"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        logger.debug("读取文件: {} | 总行数: {}", path, len(text.splitlines()))
        return "\n".join(lines)[:50000]
    except Exception as e:
        logger.error("读取文件失败: {} | 错误: {}", path, e)
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入文件，自动创建父目录"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        logger.info("写入文件: {} | 字节数: {}", path, len(content))
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        logger.error("写入文件失败: {} | 错误: {}", path, e)
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换文本（仅替换第一处匹配）"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            logger.warning("编辑失败，未找到目标文本: {} | old_text={!r}", path, old_text[:80])
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        logger.info("编辑文件: {}", path)
        return f"Edited {path}"
    except Exception as e:
        logger.error("编辑文件失败: {} | 错误: {}", path, e)
        return f"Error: {e}"


# 工具分发表：{工具名: 处理函数}
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 工具定义（API schema）
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
]
