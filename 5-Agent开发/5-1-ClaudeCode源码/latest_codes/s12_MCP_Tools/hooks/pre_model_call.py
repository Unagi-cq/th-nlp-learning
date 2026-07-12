from loguru import logger


def log_before_model_call_hook(state, model):
    logger.opt(raw=True).info(f"\033[90m[HOOK] PreModelCall: [轮次 {state.round_num}] 调用模型 {model}\033[0m\n")


HOOKS = [log_before_model_call_hook]
