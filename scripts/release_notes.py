"""Validate package/tag agreement and prepare the curated GitHub release notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_broadcast import __version__  # noqa: E402
from open_broadcast.updater import parse_version  # noqa: E402


def load_notes(version: str, tag: str = "", notes_dir: Path | None = None) -> str:
    parse_version(version)
    if tag and tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match package version {version}")
    path = (notes_dir or ROOT / "docs" / "releases") / f"{version}.md"
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith(f"# Open Broadcast {version}\n"):
        raise ValueError(f"{path} must begin with '# Open Broadcast {version}'")
    if "## " not in text:
        raise ValueError(f"{path} must include release details, not just a title")
    return text + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "release-notes.md")
    args = parser.parse_args()
    try:
        notes = load_notes(__version__, args.tag)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(notes, encoding="utf-8")
    print(f"Validated Open Broadcast {__version__} release notes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
