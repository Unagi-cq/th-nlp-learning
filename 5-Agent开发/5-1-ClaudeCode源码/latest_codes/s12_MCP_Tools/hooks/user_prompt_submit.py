from loguru import logger

from system.paths import PROJECT_ROOT


def context_inject_hook(query: str) -> str | None:
    logger.opt(raw=True).info(f"\033[90m[HOOK] UserPromptSubmit: 项目目录 {PROJECT_ROOT}\033[0m\n")
    return None


HOOKS = [context_inject_hook]
