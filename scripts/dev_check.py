"""Run local MindVault scaffold validation checks."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"==> Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode


def main() -> int:
    npm_bin = shutil.which("npm")
    if not npm_bin:
        print("Error: 'npm' executable not found on PATH.", file=sys.stderr)
        return 1

    checks = [
        ([sys.executable, "run.py", "--check"], ROOT_DIR),
        ([sys.executable, "-m", "pytest"], ROOT_DIR / "backend"),
        ([npm_bin, "run", "check"], ROOT_DIR / "frontend"),
        ([npm_bin, "test"], ROOT_DIR / "frontend"),
    ]
    for cmd, cwd in checks:
        code = _run(cmd, cwd)
        if code != 0:
            return code
    print("All MindVault scaffold checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
