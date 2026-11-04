"""Update checking against GitHub Releases, plus self-update for the Windows installer build.

Release asset naming (produced by .github/workflows/release.yml):
    OpenBroadcast-<version>-windows-x64-setup.exe
    OpenBroadcast-<version>-windows-x64-portable.zip
    OpenBroadcast-<version>-macos-arm64.dmg
    OpenBroadcast-<version>-linux-x64.tar.gz
    SHA256SUMS.txt
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from open_broadcast import __version__, paths

log = logging.getLogger(__name__)

GITHUB_REPO = os.environ.get("OPEN_BROADCAST_UPDATE_REPO", "Zigdon-Code-Factory/open-broadcast")
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
USER_AGENT = f"open-broadcast/{__version__}"
CHECKSUMS_ASSET = "SHA256SUMS.txt"

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-.]?([0-9A-Za-z.]+))?$")


def parse_version(text: str) -> tuple:
    """Parse ``v1.2.3`` / ``1.2.3-rc1`` into a sortable tuple. Pre-releases sort first."""
    m = _VERSION_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unrecognised version: {text!r}")
    major, minor, patch, pre = m.groups()
    # (1, 2, 3, 1, "") for finals; (1, 2, 3, 0, "rc1") for pre-releases.
    return (int(major), int(minor), int(patch), 0 if pre else 1, pre or "")


def is_newer(candidate: str, current: str = __version__) -> bool:
    try:
        return parse_version(candidate) > parse_version(current)
    except ValueError:
        return False


@dataclass
class Asset:
    name: str
    url: str
    size: int = 0


@dataclass
class ReleaseInfo:
    version: str
    tag: str
    html_url: str
    notes: str = ""
    assets: list[Asset] = field(default_factory=list)

    def asset(self, name: str) -> Asset | None:
        return next((a for a in self.assets if a.name == name), None)


def platform_asset_name(version: str, system: str | None = None) -> str | None:
    system = system or platform.system()
    if system == "Windows":
        return f"OpenBroadcast-{version}-windows-x64-setup.exe"
    if system == "Darwin":
        return f"OpenBroadcast-{version}-macos-arm64.dmg"
    if system == "Linux":
        return f"OpenBroadcast-{version}-linux-x64.tar.gz"
    return None


def _get(url: str, accept: str = "application/octet-stream", timeout: float = 15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return urllib.request.urlopen(req, timeout=timeout)


def parse_release(data: dict) -> ReleaseInfo:
    tag = data["tag_name"]
    return ReleaseInfo(
        version=tag.lstrip("v"),
        tag=tag,
        html_url=data.get("html_url", ""),
        notes=data.get("body") or "",
        assets=[
            Asset(a["name"], a["browser_download_url"], a.get("size", 0))
            for a in data.get("assets", [])
        ],
    )


def fetch_latest_release(url: str = API_URL) -> ReleaseInfo:
    with _get(url, accept="application/vnd.github+json") as resp:
        return parse_release(json.load(resp))


def check_for_update(current: str = __version__) -> ReleaseInfo | None:
    """Return the latest release if it's newer than ``current``, else None."""
    release = fetch_latest_release()
    if is_newer(release.version, current):
        return release
    return None


def parse_checksums(text: str) -> dict[str, str]:
    sums = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def download_asset(
    release: ReleaseInfo,
    asset: Asset,
    dest_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download ``asset`` and verify it against the release's SHA256SUMS.txt."""
    if not asset.name or asset.name in {".", ".."} or any(
        char in asset.name for char in ("/", "\\", ":")
    ):
        raise RuntimeError("Invalid release asset filename")
    sums_asset = release.asset(CHECKSUMS_ASSET)
    if sums_asset is None:
        raise RuntimeError("Release has no SHA256SUMS.txt; refusing to install unverified file")
    with _get(sums_asset.url) as resp:
        expected = parse_checksums(resp.read().decode("utf-8")).get(asset.name)
    if not expected:
        raise RuntimeError(f"No checksum listed for {asset.name}")

    dest_dir = dest_dir or Path(tempfile.mkdtemp(prefix="open-broadcast-update-"))
    dest = dest_dir / asset.name
    h = hashlib.sha256()
    try:
        with _get(asset.url, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length") or asset.size or 0)
            done = 0
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if h.hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch for {asset.name}")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


def is_installed_windows_build() -> bool:
    """True for the Inno Setup install (which can be updated in place)."""
    if platform.system() != "Windows" or not paths.is_frozen():
        return False
    app_dir = Path(sys.executable).parent
    return any(app_dir.glob("unins*.exe"))


def can_self_update() -> bool:
    return is_installed_windows_build()


def launch_installer(installer: Path) -> None:
    """Run the Inno Setup installer silently; it closes this app and relaunches it after."""
    if not can_self_update():
        raise RuntimeError("In-place updates require the Windows installer build")
    args = [
        str(installer),
        "/SILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CLOSEAPPLICATIONS",
        "/RELAUNCH=1",
        f"/DIR={Path(sys.executable).parent}",
    ]
    flags = 0
    if platform.system() == "Windows":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    log.info("Launching installer: %s", args)
    subprocess.Popen(args, creationflags=flags, close_fds=True)
