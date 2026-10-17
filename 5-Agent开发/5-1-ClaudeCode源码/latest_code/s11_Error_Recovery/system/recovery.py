import os
import random
import time
from dataclasses import dataclass

from loguru import logger

from config import MODEL

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_CONTINUATION_RETRIES = 3
MAX_TRANSIENT_RETRIES = 10
MAX_529_RETRIES_BEFORE_FALLBACK = 3
BASE_DELAY_MS = 500
MAX_DELAY_MS = 32000
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — no apology, no recap of what "
    "you were doing. Pick up mid-thought if that is where the cut happened. "
    "Break remaining work into smaller pieces."
)


@dataclass
class RecoveryState:
    current_model: str = MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    has_escalated_max_tokens: bool = False
    continuation_count: int = 0
    has_attempted_reactive_compact: bool = False
    consecutive_529: int = 0


class MaxRetriesExceeded(Exception):
    pass


def _error_text(error: Exception) -> str:
    return str(error).lower()


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int):
        return status
    text = _error_text(error)
    if "429" in text:
        return 429
    if "529" in text:
        return 529
    return None


def is_prompt_too_long_error(error: Exception) -> bool:
    text = _error_text(error)
    return "prompt_too_long" in text or "too many tokens" in text or "context_length" in text


def is_rate_limit_error(error: Exception) -> bool:
    return _status_code(error) == 429 or "rate limit" in _error_text(error)


def is_overloaded_error(error: Exception) -> bool:
    text = _error_text(error)
    return _status_code(error) == 529 or "overloaded" in text or "overload" in text


def retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response else getattr(error, "headers", None)
    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), MAX_DELAY_MS) / 1000
    return base + random.uniform(0, base * 0.25)


def maybe_switch_fallback(state: RecoveryState):
    if state.consecutive_529 < MAX_529_RETRIES_BEFORE_FALLBACK:
        return
    if not FALLBACK_MODEL or state.current_model == FALLBACK_MODEL:
        return

    state.current_model = FALLBACK_MODEL
    state.consecutive_529 = 0
    logger.warning("[recovery] 连续 529，切换备用模型: {}", FALLBACK_MODEL)


def with_retry(call, state: RecoveryState, max_retries: int = MAX_TRANSIENT_RETRIES):
    for attempt in range(max_retries):
        try:
            response = call()
            state.consecutive_529 = 0
            return response
        except Exception as error:
            if not (is_rate_limit_error(error) or is_overloaded_error(error)):
                raise

            if is_overloaded_error(error):
                state.consecutive_529 += 1
                maybe_switch_fallback(state)
            else:
                state.consecutive_529 = 0

            delay = retry_delay(attempt, retry_after_seconds(error))
            logger.warning(
                "[recovery] transient API error attempt {}/{}; sleep {:.2f}s: {}",
                attempt + 1,
                max_retries,
                delay,
                error,
            )
            time.sleep(delay)

    raise MaxRetriesExceeded(f"超过最大瞬态错误重试次数: {max_retries}")
