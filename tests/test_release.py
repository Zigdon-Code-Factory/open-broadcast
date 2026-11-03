import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from open_broadcast import __version__
from open_broadcast.updater import platform_asset_name

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_script("build")
release_notes = load_script("release_notes")


@pytest.mark.parametrize("version", ["1.2.3", "1.2.3-rc1", "1.2.3-beta.2"])
def test_windows_numeric_version_omits_prerelease(version):
    assert build.windows_version(version) == (1, 2, 3, 0)


def test_windows_version_rejects_out_of_range_components():
    with pytest.raises(ValueError, match="65535"):
        build.windows_version("65536.0.0")


def test_required_installer_cannot_silently_disappear(monkeypatch):
    monkeypatch.setattr(build, "find_iscc", lambda: None)
    with pytest.raises(RuntimeError, match="Inno Setup is required"):
        build.package_windows(require_installer=True)


def test_installer_receives_numeric_and_display_versions(monkeypatch):
    commands = []
    monkeypatch.setattr(build, "__version__", "1.2.3-rc1")
    monkeypatch.setattr(build, "find_iscc", lambda: "iscc")
    monkeypatch.setattr(build, "run", lambda command: commands.append(command))
    monkeypatch.setattr(build.shutil, "make_archive", lambda *args: "portable.zip")
    artifacts = build.package_windows(require_installer=True)
    assert "/DAppVersion=1.2.3-rc1" in commands[0]
    assert "/DAppNumericVersion=1.2.3.0" in commands[0]
    assert artifacts[-1].name == "OpenBroadcast-1.2.3-rc1-windows-x64-setup.exe"


def test_optional_installer_preserves_portable_build(monkeypatch):
    monkeypatch.setattr(build, "find_iscc", lambda: None)
    monkeypatch.setattr(build.shutil, "make_archive", lambda *args: "portable.zip")
    assert build.package_windows() == [Path("portable.zip")]


def test_curated_notes_match_current_version_and_downloads():
    text = release_notes.load_notes(__version__, f"v{__version__}")
    assert text.startswith(f"# Open Broadcast {__version__}\n")
    for system in ("Windows", "Darwin", "Linux"):
        assert platform_asset_name(__version__, system) in text
    assert f"OpenBroadcast-{__version__}-windows-x64-portable.zip" in text
    assert "SHA256SUMS.txt" in text


def test_release_tag_mismatch_fails_before_reading_notes(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        release_notes.load_notes("1.2.3", "v9.9.9", tmp_path)


def test_missing_notes_fail_release_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        release_notes.load_notes("1.2.3", "v1.2.3", tmp_path)


def test_prerelease_notes_are_supported(tmp_path):
    (tmp_path / "1.2.3-rc1.md").write_text(
        "# Open Broadcast 1.2.3-rc1\n\n## Changes\n\nRelease candidate.\n", encoding="utf-8"
    )
    assert "Release candidate." in release_notes.load_notes("1.2.3-rc1", "v1.2.3-rc1", tmp_path)


@pytest.mark.parametrize("text", ["# Wrong title\n\n## Changes\n", "# Open Broadcast 1.2.3\n"])
def test_incomplete_or_wrong_version_notes_fail(tmp_path, text):
    (tmp_path / "1.2.3.md").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        release_notes.load_notes("1.2.3", "v1.2.3", tmp_path)


def test_notes_cli_creates_preview_without_publishing(tmp_path):
    output = tmp_path / "preview.md"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/release_notes.py"), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == release_notes.load_notes(__version__)


def test_notes_cli_rejects_wrong_tag_without_writing_output(tmp_path):
    output = tmp_path / "preview.md"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/release_notes.py"), "--tag", "v999.0.0",
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not output.exists()
