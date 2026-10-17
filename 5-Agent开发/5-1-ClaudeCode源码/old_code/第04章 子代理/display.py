"""
display.py - 统一的终端输出格式化
"""
WIDTH = 120
INDENT = 2
SP = " " * INDENT


def banner(model: str, workdir: str):
    print(f"+{'=' * WIDTH}+")
    print(f"|  Agent Ready" + " " * (WIDTH - 13) + "|")
    print(f"|  cwd: {workdir}" + " " * max(1, WIDTH - len(str(workdir)) - 7) + "|")
    print(f"|  model: {model}" + " " * max(1, WIDTH - len(model) - 9) + "|")
    print(f"+{'=' * WIDTH}+")


def user_prompt():
    return input(f"\n> ")


def goodbye():
    print(f"Bye!")


def turn_header(n: int):
    print(f"\n-- Turn {n} " + "-" * (WIDTH - 8))


def tool(name: str, params: dict | None = None):
    if params:
        parts = [f"{k}={_trunc(str(v), 40)}" for k, v in params.items()]
        print(f"{SP}[{name}] {', '.join(parts)}")
    else:
        print(f"{SP}[{name}]")


def tool_output(text: str):
    preview = _trunc(text.replace("\n", " "), 200)
    if preview:
        print(f"{SP}    {preview}")


def task_delegate(desc: str, prompt: str):
    print(f"{SP}[task] {desc}")
    print(f"{SP}    prompt: {_trunc(prompt, 100)}")


def subagent_log(msg: str):
    print(f"{SP}    {msg}")


def subagent_done(turns: int, summary: str):
    print(f"{SP}    done ({turns} turns) -> {_trunc(summary, 120)}")


def agent_text(text: str):
    for line in text.split("\n"):
        print(f"{SP}{line}")


def agent_stop(reason: str):
    print(f"{SP}stop: {reason}")


def plan(items_text: str):
    print(f"\n{SP}plan:")
    for line in items_text.split("\n"):
        print(f"{SP}  {line}")


def _trunc(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s
