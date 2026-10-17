def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ {block.name} 返回大量输出")

def log_tool_use_hook(block, output):
    params = dict(block.input) if block.input else {}
    output_len = len(str(output))

    tool_name = block.name
    if tool_name == "bash":
        cmd = params.get("command", "?")
        print(f"\033[90m[HOOK] bash 完成: {cmd} → 输出 {output_len} 字符\033[0m")
    elif tool_name == "read_file":
        path = params.get("path", "?")
        limit = params.get("limit")
        limit_hint = f" (截取前 {limit} 行)" if limit else ""
        print(f"\033[90m[HOOK] read_file: {path}{limit_hint} → 输出 {output_len} 字符\033[0m")
    elif tool_name == "write_file":
        path = params.get("path", "?")
        wrote_len = len(params.get("content", ""))
        print(f"\033[90m[HOOK] write_file: {path}，写入 {wrote_len} 字节\033[0m")
    elif tool_name == "edit_file":
        path = params.get("path", "?")
        print(f"\033[90m[HOOK] edit_file: {path}，完成替换\033[0m")
    elif tool_name == "glob":
        pattern = params.get("pattern", "?")
        print(f"\033[90m[HOOK] glob: {pattern} → 输出 {output_len} 字符\033[0m")
    else:
        print(f"\033[90m[HOOK] {tool_name} 完成，输出 {output_len} 字符\033[0m")


HOOKS = [large_output_hook, log_tool_use_hook]
