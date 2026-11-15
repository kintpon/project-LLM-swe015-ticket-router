"""Tiny file logger that rolls over to the next numbered file at 300 lines."""
from pathlib import Path

MAX_LINES = 300


def log(name, message, log_dir="logs"):
    """Append one line to logs/<name>_001.log; start _002 after 300 lines."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = Path(log_dir) / f"{name}_{n:03d}.log"
        if not path.exists() or len(path.read_text(encoding="utf-8").splitlines()) < MAX_LINES:
            break
        n += 1
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
