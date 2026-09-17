"""Project-local writable paths, including resolved symlink targets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def confined(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("output path must remain inside implementation/")
    return resolved
