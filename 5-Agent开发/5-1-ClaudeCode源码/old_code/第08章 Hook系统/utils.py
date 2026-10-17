def _block_to_dict(block) -> dict | None:
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if not k.startswith("_")}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return None


def normalize_messages(messages: list) -> list:
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            blocks = []
            for block in msg["content"]:
                converted = _block_to_dict(block)
                if converted:
                    blocks.append(converted)
            clean["content"] = blocks
        else:
            clean["content"] = msg.get("content", "")
        cleaned.append(clean)

    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") not in existing_results:
                cleaned.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"], "content": "(cancelled)"}
                ]})

    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": str(prev["content"])}]
            curr_c = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged
