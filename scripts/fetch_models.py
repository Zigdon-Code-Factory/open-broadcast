"""Download all models into src/open_broadcast/models/ (bundled into release builds).

    python scripts/fetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_broadcast import models, paths  # noqa: E402


def main() -> int:
    dest_dir = paths.bundled_models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in models.MODELS.items():
        dest = dest_dir / spec.filename
        if dest.is_file() and models._sha256(dest) == spec.sha256:
            print(f"ok       {name} ({spec.filename})")
            continue
        models.download(spec, dest)
        print(f"fetched  {name} ({spec.filename})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
