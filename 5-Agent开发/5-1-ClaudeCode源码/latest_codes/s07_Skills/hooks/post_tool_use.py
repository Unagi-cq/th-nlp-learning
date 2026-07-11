from loguru import logger


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        logger.info(f"[HOOK] PostToolUse：⚠ {block.name} 返回大量输出")


def log_tool_use_hook(block, output):
    params = dict(block.input) if block.input else {}
    output_str = str(output)
    output_len = len(output_str)

    tool_name = block.name
    if tool_name == "bash":
        cmd = params.get("command", "?")
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：bash 完成: {cmd} → 输出 {output_len} 字符\033[0m\n")
    elif tool_name == "read_file":
        path = params.get("path", "?")
        limit = params.get("limit")
        limit_hint = f" (截取前 {limit} 行)" if limit else ""
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：read_file: {path}{limit_hint} → 输出 {output_len} 字符\033[0m\n")
    elif tool_name == "write_file":
        path = params.get("path", "?")
        wrote_len = len(params.get("content", ""))
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：write_file: {path}，写入 {wrote_len} 字节\033[0m\n")
    elif tool_name == "edit_file":
        path = params.get("path", "?")
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：edit_file: {path}，完成替换\033[0m\n")
    elif tool_name == "task":
        desc = params.get("description", "?")
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：task 完成: {desc} → 输出 {output_len} 字符\033[0m\n")
    elif tool_name == "glob":
        pattern = params.get("pattern", "?")
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：glob: {pattern} → 输出 {output_len} 字符\033[0m\n")
    else:
        logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：{tool_name} 完成，输出 {output_len} 字符\033[0m\n")

    # 打印输出内容（截取前 500 字符）
    preview = output_str[:500]
    if output_len > 500:
        preview += f"\n... (共 {output_len} 字符，以上为前 500 字符预览)"
    logger.opt(raw=True).info(f"\033[90m[HOOK] PostToolUse：{preview}\033[0m\n")


HOOKS = [large_output_hook, log_tool_use_hook]
