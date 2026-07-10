class State:
    """Agent 对话状态，贯穿整个 session 生命周期。"""

    def __init__(self):
        # LLM 对话历史，每条为 {"role": ..., "content": ...} 格式
        self.messages: list = []
        # 当前对话轮次编号（一次用户输入 = 一轮对话）
        self.conv_num: int = 0
        # 当前轮次内 Agent Loop 的迭代次数
        self.round_num: int = 0
        # 距上次 todo_write 调用经过的轮次数，用于触发自动 todowrite 提醒
        self.rounds_since_todo: int = 0
