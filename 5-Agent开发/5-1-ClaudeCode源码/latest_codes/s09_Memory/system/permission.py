from enum import Enum

from system.paths import PROJECT_ROOT


class PermissionMode(Enum):
    LENIENT = "lenient"      # 宽松模式：默认放行所有工具调用（硬拒绝列表除外）
    AUTO_DENY = "auto_deny"  # 所有工具调用直接拒绝
    LEARNING = "learning"    # 记住用户选择，后续同类型操作自动应用


class PermissionManager:
    """权限管理器。

    硬拒绝列表 — 始终禁止，不可绕过
    学习记忆   — 记住用户对每个工具的 allow/deny 决定
    """

    def __init__(self, mode: PermissionMode = PermissionMode.LENIENT):
        self.mode = mode
        self.workdir = PROJECT_ROOT
        # 记忆表：{context_key → "allow" | "deny"}
        self._memory: dict[str, str] = {}

        # 硬拒绝列表（始终生效，不受模式影响）
        self.deny_list = [
            "rm -rf /", "sudo", "shutdown", "reboot",
            "mkfs", "dd if=", "> /dev/sda",
        ]

    # ── 模式切换 ──────────────────────────────────────────────

    def set_mode(self, mode: PermissionMode):
        self.mode = mode

    def clear_memory(self):
        """清空学习记忆。"""
        self._memory.clear()

    # ── 上下文键 ──────────────────────────────────────────────

    def _make_context_key(self, tool_name: str, tool_input: dict) -> str:
        """为每次工具调用生成上下文键，决定记忆粒度。"""
        if tool_name == "bash":
            command = tool_input.get("command", "")
            first_word = command.strip().split()[0] if command.strip() else "empty"
            return f"bash:{first_word}"
        # 其他工具按工具名记忆
        return tool_name

    # ── 拒绝列表检查 ──────────────────────────────────────────

    def _check_deny_list(self, command: str) -> str | None:
        for pattern in self.deny_list:
            if pattern in command:
                return f"已拦截：'{pattern}' 在拒绝列表中"
        return None

    # ── 用户审批 ──────────────────────────────────────────────

    def _ask_user(self, tool_name: str, tool_input: dict) -> str:
        print(f"\n\033[33m⚠  需要确认工具调用\033[0m")
        print(f"   工具: {tool_name}({tool_input})")
        choice = input("   允许执行？[y/N] ").strip().lower()
        return "allow" if choice in ("y", "yes") else "deny"

    # ── 主入口 ────────────────────────────────────────────────

    def check_permission(self, block) -> bool:
        # 兼容 SDK ContentBlock 和 dict
        tool_input = block.input if hasattr(block, "input") else block.get("input", {})
        tool_name = block.name if hasattr(block, "name") else block.get("name", "")
        if not isinstance(tool_input, dict):
            tool_input = {}

        # 第一层：硬拒绝列表（始终生效）
        if tool_name == "bash":
            reason = self._check_deny_list(tool_input.get("command", ""))
            if reason:
                print(f"\n\033[31m⛔ {reason}\033[0m")
                return False

        ctx_key = self._make_context_key(tool_name, tool_input)

        # 宽松模式：默认放行（硬拒绝列表已在上面检查）
        if self.mode == PermissionMode.LENIENT:
            return True

        # 自动拒绝模式
        if self.mode == PermissionMode.AUTO_DENY:
            print(f"\n\033[31m⛔ 自动拒绝: {tool_name}\033[0m")
            return False

        # 学习模式
        if self.mode == PermissionMode.LEARNING:
            remembered = self._memory.get(ctx_key)
            if remembered == "allow":
                print(f"\n\033[32m✓ 已记住允许: {ctx_key}\033[0m")
                return True
            elif remembered == "deny":
                print(f"\n\033[31m⛔ 已记住拒绝: {ctx_key}\033[0m")
                return False

            # 首次遇到：询问用户并记住
            decision = self._ask_user(tool_name, tool_input)
            self._memory[ctx_key] = decision
            status = "允许" if decision == "allow" else "拒绝"
            print(f"   [学习] 已记住: {status} {ctx_key}")
            return decision == "allow"

        return True


# 全局实例，默认宽松模式
permission_manager = PermissionManager(mode=PermissionMode.LENIENT)
