"""Build a release bundle for the current platform.

    python scripts/build.py            # bundle + platform package(s) into dist/release/
    python scripts/build.py --no-package

Outputs (dist/release/):
    Windows: OpenBroadcast-<v>-windows-x64-setup.exe  (needs Inno Setup's iscc on PATH)
             OpenBroadcast-<v>-windows-x64-portable.zip
    macOS:   OpenBroadcast-<v>-macos-arm64.dmg
    Linux:   OpenBroadcast-<v>-linux-x64.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RELEASE = DIST / "release"
sys.path.insert(0, str(ROOT / "src"))

from open_broadcast import __version__  # noqa: E402
from open_broadcast.updater import parse_version  # noqa: E402


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def windows_version(version: str) -> tuple[int, int, int, int]:
    major, minor, patch = parse_version(version)[:3]
    if any(component > 65535 for component in (major, minor, patch)):
        raise ValueError("Windows version components must not exceed 65535")
    return major, minor, patch, 0


def write_windows_version_info() -> None:
    t = windows_version(__version__)
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Zigdon7'),
      StringStruct('FileDescription', 'Open Broadcast'),
      StringStruct('FileVersion', '{__version__}'),
      StringStruct('InternalName', 'OpenBroadcast'),
      StringStruct('OriginalFilename', 'OpenBroadcast.exe'),
      StringStruct('ProductName', 'Open Broadcast'),
      StringStruct('ProductVersion', '{__version__}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    (ROOT / "packaging" / "windows" / "version_info.txt").write_text(text, encoding="utf-8")


def pyinstaller() -> None:
    run([sys.executable, str(ROOT / "scripts" / "fetch_models.py")])
    if platform.system() == "Windows":
        write_windows_version_info()
    run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "open_broadcast.spec"),
         "--noconfirm", "--clean", "--distpath", str(DIST), "--workpath", str(ROOT / "build")])


def find_iscc() -> str | None:
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA") and str(Path(os.environ["LOCALAPPDATA"]) /
                                                        "Programs")):
        if base:
            p = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if p.is_file():
                return str(p)
    return None


def package_windows(require_installer: bool = False) -> list[Path]:
    iscc = find_iscc()
    if require_installer and iscc is None:
        raise RuntimeError("Inno Setup is required to build the Windows release installer")
    out = []
    bundle = DIST / "OpenBroadcast"
    zip_base = RELEASE / f"OpenBroadcast-{__version__}-windows-x64-portable"
    out.append(Path(shutil.make_archive(str(zip_base), "zip", DIST, "OpenBroadcast")))
    if iscc is None:
        print("WARNING: Inno Setup (iscc) not found; skipping installer", file=sys.stderr)
        return out
    numeric_version = ".".join(str(component) for component in windows_version(__version__))
    run([iscc, f"/DAppVersion={__version__}", f"/DAppNumericVersion={numeric_version}",
         f"/DSourceDir={bundle}",
         f"/DOutputDir={RELEASE}", str(ROOT / "packaging" / "windows" / "installer.iss")])
    out.append(RELEASE / f"OpenBroadcast-{__version__}-windows-x64-setup.exe")
    return out


def package_macos() -> list[Path]:
    app = DIST / "OpenBroadcast.app"
    # Ad-hoc sign so Apple Silicon will run it (proper Developer ID signing is optional).
    run(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    stage = DIST / "dmg"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    shutil.copytree(app, stage / app.name, symlinks=True)
    os.symlink("/Applications", stage / "Applications")
    dmg = RELEASE / f"OpenBroadcast-{__version__}-macos-arm64.dmg"
    dmg.unlink(missing_ok=True)
    run(["hdiutil", "create", "-volname", "Open Broadcast", "-srcfolder", str(stage),
         "-ov", "-format", "UDZO", str(dmg)])
    return [dmg]


def package_linux() -> list[Path]:
    tgz = RELEASE / f"OpenBroadcast-{__version__}-linux-x64.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(DIST / "OpenBroadcast", arcname=f"OpenBroadcast-{__version__}")
    return [tgz]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-package", action="store_true", help="only run PyInstaller")
    ap.add_argument("--require-installer", action="store_true",
                    help="fail if the Windows installer compiler is unavailable")
    args = ap.parse_args()

    pyinstaller()
    if args.no_package:
        return 0
    RELEASE.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system == "Windows":
        artifacts = package_windows(require_installer=args.require_installer)
    else:
        artifacts = {"Darwin": package_macos, "Linux": package_linux}[system]()
    checksums = RELEASE / "SHA256SUMS.txt"
    lines = []
    for artifact in artifacts:
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {artifact.name}\n")
    checksums.write_text("".join(lines), encoding="utf-8")
    artifacts.append(checksums)
    print("\nArtifacts:")
    for a in artifacts:
        print(f"  {a}  ({a.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
