import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config import MODEL, client
from system.paths import MEMORY_DIR, MEMORY_INDEX

MEMORY_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")

@dataclass
class Memory:
    """单条长期记忆。"""

    filename: str
    name: str
    description: str
    type: str
    body: str

    @classmethod
    def from_file(cls, path: Path, meta: dict, body: str) -> "Memory":
        return cls(
            filename=path.name,
            name=meta.get("name", path.stem),
            description=meta.get("description", ""),
            type=meta.get("type", "user"),
            body=body,
        )

    @classmethod
    def from_payload(cls, payload: dict, filename: str, allowed_types: set[str]) -> "Memory | None":
        name = str(payload.get("name") or f"memory-{int(time.time())}").strip()
        description = str(payload.get("description") or "").strip()
        body = str(payload.get("body") or "").strip()
        mem_type = str(payload.get("type") or "user").strip()

        if mem_type not in allowed_types:
            mem_type = "user"
        if not description or not body:
            return None

        return cls(
            filename=filename,
            name=name,
            description=description,
            type=mem_type,
            body=body,
        )

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "body": self.body,
        }

    def to_file_text(self) -> str:
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"type: {self.type}\n"
            f"---\n\n"
            f"{self.body}\n"
        )

    def to_display_text(self) -> str:
        return (
            f"## {self.filename}\n"
            f"name: {self.name}\n"
            f"type: {self.type}\n"
            f"description: {self.description}\n\n"
            f"{self.body}"
        )

    def index_line(self) -> str:
        description = self.description or self.body.split("\n")[0][:80]
        return f"- [{self.name}]({self.filename}) — {description}"

    def searchable_text(self) -> str:
        return " ".join([
            self.filename,
            self.name,
            self.type,
            self.description,
            self.body,
        ]).lower()

    def score(self, keywords: list[str]) -> int:
        haystack = self.searchable_text()
        return sum(1 for keyword in keywords if keyword in haystack)


class MemoryStore:
    """管理长期记忆文件、索引、检索、提取与整理。"""

    memory_types = {"user", "feedback", "project", "reference"}

    def __init__(
        self,
        memory_dir: Path = MEMORY_DIR,
        index_path: Path = MEMORY_INDEX,
        model: str = MODEL,
        llm_client=client,
        consolidate_threshold: int = 10,
    ):
        self.memory_dir = memory_dir
        self.index_path = index_path
        self.model = model
        self.client = llm_client
        self.consolidate_threshold = consolidate_threshold

    @staticmethod
    def extract_text(content) -> str:
        if not isinstance(content, list):
            return str(content)
        return "\n".join(
            getattr(block, "text", "")
            for block in content
            if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        if not text.startswith("---"):
            return {}, text

        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text

        meta = {}
        for line in parts[1].strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, parts[2].strip()

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        compact = query.strip().lower()
        terms = [term for term in compact.split() if term]
        if compact and compact not in terms:
            terms.append(compact)
        return terms

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", name.strip().lower())
        slug = re.sub(r"-+", "-", slug).strip("-_")
        return slug or f"memory-{int(time.time())}"

    @staticmethod
    def _parse_json_array(text: str) -> list:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("[memory] JSON 解析失败: {}", exc)
            return []
        return data if isinstance(data, list) else []

    def rebuild_index(self):
        """从所有记忆文件重建 MEMORY.md 索引。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        lines = [memory.index_line() for memory in self.list_memories()]
        self.index_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def write(self, name: str, mem_type: str, description: str, body: str):
        """写入单个记忆文件，含 YAML frontmatter。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        safe_type = mem_type if mem_type in self.memory_types else "user"
        memory = Memory(
            filename=f"{self._slugify(name)}.md",
            name=name,
            description=description,
            type=safe_type,
            body=body,
        )
        filepath = self.memory_dir / memory.filename
        filepath.write_text(memory.to_file_text(), encoding="utf-8")
        self.rebuild_index()
        return filepath

    def read_index(self) -> str:
        """读取 MEMORY.md 索引，供 system prompt 展示可查询目录。"""
        if not self.index_path.exists():
            return ""
        return self.index_path.read_text(encoding="utf-8").strip()

    def list_memories(self) -> list[Memory]:
        """列出所有记忆对象。"""
        if not self.memory_dir.exists():
            return []

        result = []
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == self.index_path.name:
                continue
            raw = path.read_text(encoding="utf-8")
            meta, body = self.parse_frontmatter(raw)
            result.append(Memory.from_file(path, meta, body))
        return result

    def list_files(self) -> list[dict]:
        """列出所有记忆文件及其元数据，保留 dict 输出方便工具层使用。"""
        return [memory.to_dict() for memory in self.list_memories()]

    def query(self, query: str = "", filenames: list[str] | None = None, max_items: int = 5) -> str:
        """查询或读取记忆，供 query_memory 工具调用。"""
        max_items = max(1, min(max_items or 5, 10))
        memories = self.list_memories()
        if not memories:
            return "暂无记忆。"

        if filenames:
            by_filename = {memory.filename: memory for memory in memories}
            parts = []
            missing = []
            for filename in filenames[:max_items]:
                safe_name = filename.strip().split("/")[-1]
                memory = by_filename.get(safe_name)
                if not memory:
                    missing.append(filename)
                    continue
                parts.append(f"## {safe_name}\n{memory.to_file_text()}")
            if missing:
                parts.append("未找到记忆文件: " + ", ".join(missing))
            return "\n\n".join(parts) if parts else "未找到匹配的记忆文件。"

        query = query.strip()
        if not query:
            index = self.read_index()
            return f"可用记忆索引:\n{index}" if index else "暂无记忆索引。"

        keywords = self._query_terms(query)
        if not keywords:
            return "请提供有效的查询关键词。"

        ranked = []
        for memory in memories:
            score = memory.score(keywords)
            if score > 0:
                ranked.append((score, memory))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].filename))

        selected = [memory for _, memory in ranked[:max_items]]
        if not selected:
            return "未找到相关记忆。可先不带参数调用 query_memory 查看记忆索引。"

        return "\n\n".join(memory.to_display_text() for memory in selected)

    def extract_from_messages(self, messages: list):
        """从最近对话中提取新记忆。每轮结束后运行。"""

        def _format_dialogue(messages: list) -> str:
            """提取并构造纯文本的对话记录"""
            parts = []
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = self.extract_text(content)
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)

        dialogue = _format_dialogue(messages)
        if not dialogue.strip():
            return

        existing = self.list_memories()
        existing_desc = "\n".join(
            f"- {memory.name}: {memory.description}" for memory in existing
        ) if existing else "(none)"

        prompt = (
            "Extract user preferences, constraints, or project facts from this dialogue.\n"
            "Return a JSON array. Each item: {name, type, description, body}.\n"
            "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
            "- type: one of 'user' (user preference), 'feedback' (guidance), "
            "'project' (project fact), 'reference' (external pointer)\n"
            "- description: one-line summary for index lookup\n"
            "- body: full detail in markdown\n"
            "If nothing new or already covered by existing memories, return [].\n\n"
            f"Existing memories:\n{existing_desc}\n\n"
            f"Dialogue:\n{dialogue[:4000]}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )
        except Exception as exc:
            logger.warning("[memory] 提取记忆失败: {}", exc)
            return

        count = 0
        for raw_item in self._parse_json_array(self.extract_text(response.content).strip()):
            if not isinstance(raw_item, dict):
                continue
            filename = f"{self._slugify(str(raw_item.get('name') or 'memory'))}.md"
            memory = Memory.from_payload(raw_item, filename, self.memory_types)
            if not memory:
                continue
            self.write(memory.name, memory.type, memory.description, memory.body)
            count += 1

        if count:
            logger.info("[memory] 提取了 {} 条新记忆", count)

    def consolidate(self):
        """合并重复/过时记忆。文件数达到阈值后触发。"""
        memories = self.list_memories()
        if len(memories) < self.consolidate_threshold:
            return

        catalog = "\n\n".join(
            memory.to_display_text()
            for memory in memories
        )
        prompt = (
            "Consolidate the following memory files. Rules:\n"
            "1. Merge duplicates into one\n"
            "2. Remove outdated/contradicted memories\n"
            "3. Keep the total under 30 memories\n"
            "4. Preserve important user preferences above all\n"
            "Return a JSON array. Each item: {name, type, description, body}.\n\n"
            f"{catalog[:16000]}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
        except Exception as exc:
            logger.warning("[memory] 整理记忆失败: {}", exc)
            return

        items = []
        for raw_item in self._parse_json_array(self.extract_text(response.content).strip()):
            if isinstance(raw_item, dict):
                filename = f"{self._slugify(str(raw_item.get('name') or 'memory'))}.md"
                memory = Memory.from_payload(raw_item, filename, self.memory_types)
                if memory:
                    items.append(memory)
        if not items:
            return

        for path in self.memory_dir.glob("*.md"):
            if path.name != self.index_path.name:
                path.unlink()

        for memory in items:
            self.write(memory.name, memory.type, memory.description, memory.body)

        logger.info("[memory] 合并整理 {} → {} 条记忆", len(memories), len(items))


memory_store = MemoryStore()


def query_memory(query: str = "", filenames: list[str] | None = None, max_items: int = 5) -> str:
    return memory_store.query(query=query, filenames=filenames, max_items=max_items)


def schedule_memory_maintenance(messages: list[dict]):
    def _run_memory_maintenance(messages_snapshot: list[dict]):
        try:
            memory_store.extract_from_messages(messages_snapshot)
            memory_store.consolidate()
        except Exception as exc:
            logger.exception("[memory] 后台维护失败: {}", exc)

    pre_compress = [
        msg if isinstance(msg, dict) else {
            "role": getattr(msg, "role", ""),
            "content": str(getattr(msg, "content", "")),
        }
        for msg in messages
    ]
    MEMORY_WORKER.submit(_run_memory_maintenance, pre_compress)
    logger.debug("[memory] 已提交后台记忆维护任务")
