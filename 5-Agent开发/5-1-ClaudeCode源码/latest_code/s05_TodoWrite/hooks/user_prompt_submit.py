from pathlib import Path

WORKDIR = Path.cwd()


def context_inject_hook(query: str) -> str | None:
    print(f"\033[90m[HOOK] UserPromptSubmit: 当前工作目录 {WORKDIR}\033[0m")
    return None


HOOKS = [context_inject_hook]
