"""
file.py - 文件操作工具：bash, read_file, write_file, edit_file
"""
import subprocess
from pathlib import Path

from loguru import logger

WORKDIR = Path.cwd()

def _has_write_side_effect(command: str) -> bool:
    """检测 bash 命令中是否有写入副作用。"""
    # 输出/追加重定向（排除 << heredoc 但保守策略不排除）
    if ">" in command:
        return True
    # tee 写文件（不管有没有 | tee，tee 本身就会写）
    if "tee " in command or command.endswith("tee"):
        return True
    # sed -i / awk -i inplace 原地编辑
    if "sed " in command and " -i" in command:
        return True
    # 追加重定向 >> 已被 > 覆盖
    return False


def bash_is_concurrency_safe(command: str) -> bool:
    """判断 bash 命令在此输入下能否并发。
    两步：1) 检测写入副作用 2) 白名单命令字。
    保守策略：不在白名单或有写入副作用的命令一律视为 unsafe。"""
    if _has_write_side_effect(command):
        return False
    cmd = command.strip().split()[0] if command.strip() else ""
    return cmd in READONLY_COMMANDS


READONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "find", "grep", "wc", "du", "df",
    "echo", "printf", "pwd", "which", "whoami", "date", "uname",
    "env", "printenv", "stat", "file", "tree", "sort", "uniq",
    "cut", "tr", "column", "diff", "comm", "cmp", "dirname",
    "basename", "realpath", "readlink",
}


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        logger.warning("路径逃逸工作目录: {}", p)
        raise ValueError(f"路径逃逸工作目录: {p}")
    return path


def run_bash(command: str) -> str:
    deny_list = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]
    if any(d in command for d in deny_list):
        logger.warning("[bash] 危险命令已拦截: {}", command)
        return "错误：危险命令已拦截"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "（无输出）"
    except subprocess.TimeoutExpired:
        logger.error("[bash] 超时 (120s): {}", command[:60])
        return "错误：超时 (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        total = len(lines)
        if limit and limit < total:
            lines = lines[:limit] + [f"... （还有 {total - limit} 行）"]
        return "\n".join(lines)
    except Exception as e:
        logger.error("[read] {} | {}", path, e)
        return f"错误：{e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"已写入 {len(content)} 字节到 {path}"
    except Exception as e:
        logger.error("[write] {} | {}", path, e)
        return f"错误：{e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            logger.warning("[edit] 未找到目标文本: {}", path)
            return f"错误：在 {path} 中未找到目标文本"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"已编辑 {path}"
    except Exception as e:
        logger.error("[edit] {} | {}", path, e)
        return f"错误：{e}"


def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "（无匹配）"
    except Exception as e:
        return f"错误：{e}"


TOOLS = [
    {
        "name": "bash",
        "description": "执行 shell 命令。",
        "handler": run_bash,
        "concurrency_safe": False,
        "concurrency_check": lambda command: bash_is_concurrency_safe(command),
        "isReadOnly": False,
        "readonly_check": lambda command: bash_is_concurrency_safe(command),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "读取文件内容。",
        "handler": run_read,
        "concurrency_safe": True,
        "maxResultSizeChars": float("inf"),
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
        "description": "写入内容到文件。",
        "handler": run_write,
        "concurrency_safe": False,
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
        "description": "替换文件中的精确文本一次。",
        "handler": run_edit,
        "concurrency_safe": False,
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
        "name": "glob",
        "description": "查找匹配 glob 模式的文件。",
        "handler": run_glob,
        "concurrency_safe": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
]
