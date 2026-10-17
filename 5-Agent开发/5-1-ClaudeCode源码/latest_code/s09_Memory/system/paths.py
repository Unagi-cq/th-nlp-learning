from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = PROJECT_ROOT / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
SKILLS_DIR = PROJECT_ROOT / "skill-files"
SPILL_DIR = PROJECT_ROOT / ".tool_results"
TRANSCRIPT_DIR = PROJECT_ROOT / ".transcripts"


def resolve_in_project(path: str) -> Path:
    """Resolve a user supplied path and keep it inside this example project."""
    target = (PROJECT_ROOT / path).resolve()
    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"路径超出项目目录: {path}")
    return target
