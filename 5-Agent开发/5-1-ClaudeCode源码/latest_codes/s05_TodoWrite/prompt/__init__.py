from datetime import datetime
from pathlib import Path

WORKDIR = Path.cwd()

_TEMPLATE = (Path(__file__).parent / "system.txt").read_text()


def load_system_prompt() -> str:
    return _TEMPLATE.format(
        pwd=WORKDIR,
        current_date_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
