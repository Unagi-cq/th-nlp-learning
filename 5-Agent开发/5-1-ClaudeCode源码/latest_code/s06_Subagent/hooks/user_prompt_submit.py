from pathlib import Path

from loguru import logger

WORKDIR = Path.cwd()


def context_inject_hook(query: str) -> str | None:
    logger.opt(raw=True).info(f"\033[90m[HOOK] UserPromptSubmit: 当前工作目录 {WORKDIR}\033[0m\n")
    return None


HOOKS = [context_inject_hook]
