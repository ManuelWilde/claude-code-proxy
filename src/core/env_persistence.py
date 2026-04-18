import os
from pathlib import Path
from typing import Dict, Optional


def read_env(path: str) -> Dict[str, str]:
    """Read a .env file into a dict, ignoring comments and blank lines."""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = _unquote(value.strip())
    return result


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _quote(value: str) -> str:
    if not value or " " in value or "#" in value or '"' in value:
        return f'"{value}"'
    return value


def update_env(path: str, updates: Dict[str, str]):
    """Update a .env file, preserving comments and structure.

    - Existing keys are updated in-place.
    - New keys are appended at the end.
    - Comments and blank lines are preserved.
    """
    lines = []
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()

    remaining = dict(updates)
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            key, _, _ = stripped.partition("=")
            key = key.strip()
            if key in remaining:
                new_lines.append(f"{key}={_quote(remaining[key])}\n")
                del remaining[key]
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if remaining:
        new_lines.append("\n# Updated by dashboard\n")
        for key, value in remaining.items():
            new_lines.append(f"{key}={_quote(value)}\n")

    with open(path, "w") as f:
        f.writelines(new_lines)


def get_env_path() -> str:
    """Find the .env file path by walking up to the project root."""
    dir_path = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = dir_path / ".env"
        if candidate.exists():
            return str(candidate)
        parent = dir_path.parent
        if parent == dir_path:
            break
        dir_path = parent
    # Fallback: project root (3 levels up from src/core/env_persistence.py)
    return str(Path(__file__).resolve().parent.parent.parent / ".env")
