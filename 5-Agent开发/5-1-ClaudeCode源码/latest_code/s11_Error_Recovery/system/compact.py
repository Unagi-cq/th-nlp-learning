"""
Context Compaction — 分层压缩对话历史，避免超出模型上下文窗口。

两层策略（按优先级递增，后层在前层基础上进一步压缩）：
  L1 snipCompact   — 消息数超限时，裁剪中间消息，保留头尾
  L2 autoCompact   — LLM 对话摘要，将完整历史压缩为一段摘要文本
  Emergency reactiveCompact — API 报上下文超限错误时的紧急压缩
"""
import json
import time
from pathlib import Path

from loguru import logger

from config import MODEL, client
from system.paths import TRANSCRIPT_DIR

CONTEXT_LIMIT = 50000       # 上下文窗口字节上限（触发 L2 autoCompact）

def estimate_size(msgs: list) -> int:
    """估算消息列表的字符串长度。"""
    return len(str(msgs))


def _block_type(block) -> str | None:
    """统一获取 block 的 type 字段（兼容 dict 和 object 两种格式）。"""
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _message_has_tool_use(msg: dict) -> bool:
    """判断一条 assistant 消息是否包含 tool_use block。"""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg: dict) -> bool:
    """判断一条 user 消息是否为 tool_result 消息。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


# ---------------------------------------------------------------------------
# L1: snipCompact — 消息数裁剪
# ---------------------------------------------------------------------------

def snip_compact(messages: list, max_messages: int = 100) -> list:
    """消息数量超限时，保留头尾，中间裁剪并用占位消息替代。

    边界保护：裁剪点不会切断 tool_use ↔ tool_result 的配对关系。
    """
    if len(messages) <= max_messages:
        return messages

    head_end, tail_start = 3, len(messages) - (max_messages - 3)

    # 如果裁剪起点前一条是 tool_use，向后跳过紧随的 tool_result
    if _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # 如果裁剪终点是一条 tool_result 且前一条是 tool_use，向前吞并 tool_use
    if (0 < tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1

    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end
    return (
        messages[:head_end]
        + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
        + messages[tail_start:]
    )


# ---------------------------------------------------------------------------
# L2: autoCompact — LLM 对话摘要
# ---------------------------------------------------------------------------

def write_transcript(messages: list) -> Path:
    """将当前对话历史写入 JSONL 转录文件，供调试和恢复使用。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    return path


def summarize_history(messages: list, focus: str = "") -> str:
    """调用 LLM 将对话历史压缩为一段紧凑摘要。

    保留信息：当前目标、关键发现/决策、已读/改文件、剩余工作、用户约束。
    """
    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:80000]
    focus_line = f"Focus especially on: {focus}\n" if focus else ""
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
        "4. remaining work, 5. user constraints.\n"
        + focus_line +
        "Be compact but concrete.\n\n"
        + conversation
    )
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip() or "(empty summary)"


def compact_history(messages: list, focus: str = "") -> list:
    """L2 入口：先转录落盘，再 LLM 摘要，返回单条压缩消息。"""
    transcript_path = write_transcript(messages)
    logger.info("[compact] transcript saved: {}", transcript_path)
    summary = summarize_history(messages, focus)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ---------------------------------------------------------------------------
# Emergency: reactiveCompact — API 报错时紧急压缩
# ---------------------------------------------------------------------------

def reactive_compact(messages: list) -> list:
    """API 返回上下文超限错误时触发，摘要全文 + 保留最近 5 条消息的尾部。

    与 compact_history 的区别：保留尾部使 Agent 能从断点继续。
    """
    transcript = write_transcript(messages)
    logger.warning("[compact] reactive compact triggered, transcript: {}", transcript)
    summary = summarize_history(messages)

    # 保留尾部最近几条消息，不断开 tool_use ↔ tool_result 配对
    tail_start = max(0, len(messages) - 5)
    if (0 < tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1

    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
        *messages[tail_start:],
    ]
