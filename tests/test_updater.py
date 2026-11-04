import hashlib
import io

import pytest

from open_broadcast import updater


@pytest.mark.parametrize(
    "a,b",
    [
        ("0.1.1", "0.1.0"),
        ("v0.2.0", "0.1.9"),
        ("1.0.0", "1.0.0-rc1"),
        ("1.0.0-rc2", "1.0.0-rc1"),
        ("10.0.0", "9.9.9"),
    ],
)
def test_is_newer(a, b):
    assert updater.is_newer(a, b)
    assert not updater.is_newer(b, a)


def test_same_version_not_newer():
    assert not updater.is_newer("0.1.0", "0.1.0")


def test_garbage_version_not_newer():
    assert not updater.is_newer("nightly", "0.1.0")


def test_parse_release_and_assets():
    data = {
        "tag_name": "v1.2.3",
        "html_url": "https://example.com/r",
        "body": "notes",
        "assets": [
            {"name": "OpenBroadcast-1.2.3-windows-x64-setup.exe",
             "browser_download_url": "https://x/setup.exe", "size": 10},
            {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums", "size": 1},
        ],
    }
    r = updater.parse_release(data)
    assert r.version == "1.2.3"
    name = updater.platform_asset_name(r.version, "Windows")
    assert r.asset(name).url == "https://x/setup.exe"
    assert r.asset("missing") is None


def test_platform_asset_names():
    assert updater.platform_asset_name("1.0.0", "Darwin").endswith("macos-arm64.dmg")
    assert updater.platform_asset_name("1.0.0", "Linux").endswith("linux-x64.tar.gz")
    assert updater.platform_asset_name("1.0.0", "Plan9") is None


def test_parse_checksums():
    text = "abc123  file-a.exe\nDEF456 *file-b.zip\n\n"
    assert updater.parse_checksums(text) == {"file-a.exe": "abc123", "file-b.zip": "def456"}


def test_not_self_updatable_from_source():
    assert updater.can_self_update() is False


def test_installed_windows_build_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updater.paths, "is_frozen", lambda: True)
    monkeypatch.setattr(updater.sys, "executable", str(tmp_path / "OpenBroadcast.exe"))
    assert not updater.can_self_update()
    (tmp_path / "unins000.exe").touch()
    assert updater.can_self_update()


def test_installer_targets_existing_install_and_relaunches(monkeypatch, tmp_path):
    install_dir = tmp_path / "Existing install"
    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updater.sys, "executable", str(install_dir / "OpenBroadcast.exe"))
    monkeypatch.setattr(updater.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(updater.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    launches = []
    monkeypatch.setattr(
        updater.subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs))
    )
    installer = tmp_path / "setup.exe"
    updater.launch_installer(installer)
    command, options = launches[0]
    assert command == [
        str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS",
        "/RELAUNCH=1", f"/DIR={install_dir}",
    ]
    assert options == {"creationflags": 0x208, "close_fds": True}


def test_installer_cannot_run_from_unsupported_build(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "can_self_update", lambda: False)
    with pytest.raises(RuntimeError, match="Windows installer build"):
        updater.launch_installer(tmp_path / "setup.exe")


@pytest.fixture
def download_release(monkeypatch):
    payload = b"test installer payload"
    asset = updater.Asset("setup.exe", "https://example.com/setup.exe", len(payload))
    sums = updater.Asset("SHA256SUMS.txt", "https://example.com/sums")
    release = updater.ReleaseInfo("1.0.0", "v1.0.0", "https://example.com", assets=[asset, sums])
    checksum = hashlib.sha256(payload).hexdigest()

    def fake_get(url, **kwargs):
        response = io.BytesIO(
            f"{checksum}  {asset.name}\n".encode() if url == sums.url else payload
        )
        response.headers = {}
        return response

    monkeypatch.setattr(updater, "_get", fake_get)
    return release, asset, payload


def test_download_verifies_checksum(download_release, tmp_path):
    release, asset, payload = download_release
    progress = []
    result = updater.download_asset(
        release, asset, tmp_path, progress=lambda done, total: progress.append((done, total))
    )
    assert result.read_bytes() == payload
    assert progress == [(len(payload), len(payload))]


def test_download_rejects_tampering(download_release, monkeypatch, tmp_path):
    release, asset, payload = download_release
    original_get = updater._get

    def tampered_get(url, **kwargs):
        response = original_get(url, **kwargs)
        if url == asset.url:
            response.seek(0)
            response.write(b"tampered")
            response.seek(0)
        return response

    monkeypatch.setattr(updater, "_get", tampered_get)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        updater.download_asset(release, asset, tmp_path)
    assert not (tmp_path / asset.name).exists()


def test_cancelled_download_removes_partial_file(download_release, tmp_path):
    release, asset, payload = download_release

    def cancel(done, total):
        raise RuntimeError("Cancelled")

    with pytest.raises(RuntimeError, match="Cancelled"):
        updater.download_asset(release, asset, tmp_path, progress=cancel)
    assert not (tmp_path / asset.name).exists()


def test_download_requires_checksums(download_release, tmp_path):
    release, asset, payload = download_release
    release.assets = [asset]
    with pytest.raises(RuntimeError, match="no SHA256SUMS"):
        updater.download_asset(release, asset, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["../setup.exe", r"..\setup.exe", "C:setup.exe", "", ".."])
def test_download_rejects_unsafe_names(download_release, tmp_path, name):
    release, asset, payload = download_release
    asset.name = name
    with pytest.raises(RuntimeError, match="Invalid release asset filename"):
        updater.download_asset(release, asset, tmp_path)
    assert not list(tmp_path.iterdir())
