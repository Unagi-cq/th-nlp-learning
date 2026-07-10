import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from hooks import hook_registry

WORKDIR = Path.cwd()
SPILL_DIR = WORKDIR / ".tool_results"

DEFAULT_MAX_RESULT_SIZE = 50_000

def maybe_spill(tool_name: str, output: str, max_size: float) -> str:
    """结果超过 max_size 则落盘，返回预览 + 文件路径；否则原样返回。"""
    if len(output) <= max_size:
        return output

    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    spill_path = SPILL_DIR / f"{tool_name}_{ts}.txt"
    spill_path.write_text(output)

    preview = output[:2000]
    hint = f"\n\n（完整结果已保存到 {spill_path}，共 {len(output)} 字符，以上为前 2000 字符预览）"
    logger.info("[spill] {} 结果 {} 字符 → {}", tool_name, len(output), spill_path)
    return preview + hint


class Tool:
    """工具定义：从模块的 dict 构造，剥离框架字段后提供 API dict。"""

    def __init__(self, raw: dict):
        self.name = raw["name"]
        self.description = raw["description"]
        self.input_schema = raw["input_schema"]
        self.handler: Callable = raw["handler"]
        self.max_result_size: float = raw.get("maxResultSizeChars", DEFAULT_MAX_RESULT_SIZE)

        self._concurrency_check: Callable | None = raw.get("concurrency_check")
        self.concurrency_safe: bool = raw.get("concurrency_safe", True)
        self._readonly_check: Callable | None = raw.get("readonly_check")
        self.is_readonly: bool = raw.get("isReadOnly", self.concurrency_safe)

    def is_concurrency_safe_for_input(self, **params) -> bool:
        """判断当前输入下能否并发。优先用 per-input check，其次是工具级默认值。"""
        if self._concurrency_check:
            return self._concurrency_check(**params)
        return self.concurrency_safe

    def is_readonly_for_input(self, **params) -> bool:
        """判断当前输入是否只读。优先用 per-input check，其次是工具级默认值。"""
        if self._readonly_check:
            return self._readonly_check(**params)
        return self.is_readonly

    def to_api_dict(self) -> dict:
        """返回发给模型 API 的 dict，不包含 handler、maxResultSizeChars 等框架字段。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """工具注册中心：管理所有工具的定义、执行、落盘。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def to_api_list(self) -> list[dict]:
        """返回模型 API 所需的工具定义列表。"""
        return [t.to_api_dict() for t in self._tools.values()]

    def execute(self, name: str, **params) -> str:
        """执行工具并自动落盘。"""
        tool = self._tools[name]
        output = tool.handler(**params)
        return maybe_spill(name, output, tool.max_result_size)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


MAX_CONCURRENCY = 10


def partition_tool_calls(tool_use_blocks):
    """把工具调用按连续并发安全块分批。
    [read A, read B, glob *.py, bash "rm x", read C]
      → batch1(并发): [read A, read B, glob *.py]
      → batch2(串行): [bash "rm x"]
      → batch3(并发): [read C]
    """
    batches = []
    current_batch = []

    for block in tool_use_blocks:
        tool = tool_registry.get(block.name)
        params = dict(block.input) if block.input else {}

        if tool and tool.is_concurrency_safe_for_input(**params):
            current_batch.append(block)
        else:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([block])

    if current_batch:
        batches.append(current_batch)

    return batches


def _run_one_tool(block):
    """执行单个工具调用，异常时返回错误文本而非抛出。"""
    params = dict(block.input) if block.input else {}
    try:
        output = tool_registry.execute(block.name, **params)
        hook_registry.trigger("PostToolUse", block, output)
        return output
    except Exception as e:
        logger.error(f"[{block.name}] 工具执行异常: {e}")
        return f"错误：{block.name} 执行失败 - {e}"


def execute_batch(batch):
    """并发执行一个 batch 内的所有工具调用。单元素直接执行，多元素线程池并发。"""
    results = []

    if len(batch) == 1:
        block = batch[0]
        output = _run_one_tool(block)
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
        return results

    with ThreadPoolExecutor(max_workers=min(len(batch), MAX_CONCURRENCY)) as executor:
        # 全部提交（并行执行），按原始顺序收集结果
        futures = [executor.submit(_run_one_tool, block) for block in batch]
        for block, future in zip(batch, futures):
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": future.result(),
            })

    return results

# 自动扫描所有子模块，将 dict 转为 Tool 并注册
tool_registry = ToolRegistry()

for mod_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    mod = importlib.import_module(mod_info.name)
    if hasattr(mod, "TOOLS"):
        for raw in mod.TOOLS:
            tool_registry.register(Tool(raw))
