import os
from pathlib import Path

from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv

# ########## 模型与日志配置
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

logger.add("../app.log", rotation="5 MB", encoding="utf-8", level="DEBUG")

client = Anthropic(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

MODEL = os.getenv("MODEL_NAME")