from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication

from open_broadcast import updater
from open_broadcast.config import PipelineConfig
from open_broadcast.ui import tray, update
from open_broadcast.ui.update import UpdateController


@pytest.fixture(scope="session")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(application):
    manager = SimpleNamespace(
        config=PipelineConfig(check_for_updates=False), stop=Mock(), update_config=Mock()
    )
    instance = UpdateController(manager)
    yield instance
    instance._timer.stop()
    assert application is not None


@pytest.fixture
def synchronous_threads(monkeypatch):
    monkeypatch.setattr(
        update.threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target)
    )


@pytest.fixture
def release():
    return updater.ReleaseInfo("0.1.1", "v0.1.1", "https://example.com/releases/v0.1.1")


def test_disabled_background_check_does_not_start(controller):
    controller.check(interactive=False)
    assert not controller._busy


def test_disabled_background_result_does_not_prompt(controller, monkeypatch):
    def unexpected_prompt(release):
        pytest.fail("Disabled update check must not prompt")

    monkeypatch.setattr(controller, "_prompt", unexpected_prompt)
    controller._busy = True
    controller._on_checked(SimpleNamespace(version="9.0.0"), None, False)
    assert not controller._busy


def test_busy_download_does_not_start_another(controller):
    controller._busy = True
    controller._download_and_install(None)
    assert controller._busy


def test_cancelled_download_clears_busy_state(controller):
    controller._busy = True
    controller._on_downloaded(None, "Cancelled")
    assert not controller._busy


def test_one_click_checks_and_installs_without_prompt(
    controller, release, synchronous_threads, monkeypatch
):
    monkeypatch.setattr(updater, "check_for_update", lambda: release)
    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    download = Mock()
    prompt = Mock()
    monkeypatch.setattr(controller, "_download_and_install", download)
    monkeypatch.setattr(controller, "_prompt", prompt)
    controller.manager.config.skipped_version = release.version
    controller.update_now()
    download.assert_called_once_with(release)
    prompt.assert_not_called()


def test_one_click_completes_download_and_installer_handoff(
    controller, release, synchronous_threads, monkeypatch, tmp_path
):
    name = updater.platform_asset_name(release.version)
    release.assets = [updater.Asset(name, "https://example.com/installer")]
    installer = tmp_path / name

    def download(release_info, asset, progress):
        assert release_info is release
        assert asset is release.assets[0]
        installer.write_bytes(b"verified installer")
        progress(1, 1)
        return installer

    monkeypatch.setattr(updater, "check_for_update", lambda: release)
    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    monkeypatch.setattr(updater, "download_asset", download)
    launch = Mock()
    quit_app = Mock()
    monkeypatch.setattr(updater, "launch_installer", launch)
    monkeypatch.setattr(update.QApplication, "quit", quit_app)
    controller.update_now()
    launch.assert_called_once_with(installer)
    controller.manager.stop.assert_called_once()
    quit_app.assert_called_once()
    assert not controller._busy
    assert controller._progress_dialog is None
    assert controller._download_cancelled is None


@pytest.mark.parametrize("install", [False, True])
def test_background_check_never_installs_without_confirmation(
    controller, release, synchronous_threads, monkeypatch, install
):
    controller.manager.config.check_for_updates = True
    monkeypatch.setattr(updater, "check_for_update", lambda: release)
    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    download = Mock()
    prompt = Mock()
    monkeypatch.setattr(controller, "_download_and_install", download)
    monkeypatch.setattr(controller, "_prompt", prompt)
    controller.check(interactive=False, install=install)
    download.assert_not_called()
    prompt.assert_called_once_with(release)


@pytest.mark.parametrize("self_update,install", [(True, False), (False, True)])
def test_check_only_and_unsupported_builds_keep_confirmation_flow(
    controller, release, synchronous_threads, monkeypatch, self_update, install
):
    monkeypatch.setattr(updater, "check_for_update", lambda: release)
    monkeypatch.setattr(updater, "can_self_update", lambda: self_update)
    download = Mock()
    prompt = Mock()
    monkeypatch.setattr(controller, "_download_and_install", download)
    monkeypatch.setattr(controller, "_prompt", prompt)
    controller.check(interactive=True, install=install)
    download.assert_not_called()
    prompt.assert_called_once_with(release)


def test_one_click_up_to_date_clears_busy_state(
    controller, synchronous_threads, monkeypatch
):
    monkeypatch.setattr(updater, "check_for_update", lambda: None)
    information = Mock()
    monkeypatch.setattr(update.QMessageBox, "information", information)
    busy_states = []
    controller.busy_changed.connect(busy_states.append)
    controller.update_now()
    information.assert_called_once()
    assert busy_states == [True, False]
    assert not controller._busy
    controller.manager.stop.assert_not_called()


def test_one_click_network_failure_keeps_app_running(
    controller, synchronous_threads, monkeypatch
):
    monkeypatch.setattr(updater, "check_for_update", Mock(side_effect=OSError("Offline")))
    warning = Mock()
    monkeypatch.setattr(update.QMessageBox, "warning", warning)
    controller.update_now()
    warning.assert_called_once()
    assert "Offline" in warning.call_args.args[-1]
    assert not controller._busy
    controller.manager.stop.assert_not_called()


def test_update_click_ignored_while_busy(controller, monkeypatch):
    check = Mock()
    monkeypatch.setattr(updater, "check_for_update", check)
    controller._set_busy(True)
    controller.update_now()
    check.assert_not_called()


def test_failed_download_does_not_stop_or_install(controller, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(updater, "launch_installer", launch)
    monkeypatch.setattr(update.QMessageBox, "warning", Mock())
    controller._set_busy(True)
    controller._on_downloaded(None, "Checksum mismatch")
    controller.manager.stop.assert_not_called()
    launch.assert_not_called()
    assert not controller._busy


def test_verified_download_stops_installs_and_quits(controller, monkeypatch, tmp_path):
    actions = []
    installer = tmp_path / "setup.exe"
    controller.manager.stop = lambda: actions.append("stop")
    monkeypatch.setattr(updater, "launch_installer", lambda path: actions.append(path))
    monkeypatch.setattr(update.QApplication, "quit", lambda: actions.append("quit"))
    controller._on_downloaded(installer, None)
    assert actions == ["stop", installer, "quit"]


def test_cancel_after_download_finishes_still_prevents_install(controller, monkeypatch, tmp_path):
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"downloaded installer")
    launch = Mock()
    monkeypatch.setattr(updater, "launch_installer", launch)
    controller._download_cancelled = update.threading.Event()
    controller._download_cancelled.set()
    controller._set_busy(True)
    controller._on_downloaded(installer, None)
    launch.assert_not_called()
    controller.manager.stop.assert_not_called()
    assert not installer.exists()
    assert not controller._busy
    assert controller._download_cancelled is None


def test_installer_launch_failure_does_not_quit_app(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "launch_installer", Mock(side_effect=OSError("Denied")))
    warning = Mock()
    quit_app = Mock()
    monkeypatch.setattr(update.QMessageBox, "warning", warning)
    monkeypatch.setattr(update.QApplication, "quit", quit_app)
    controller._on_downloaded(tmp_path / "setup.exe", None)
    warning.assert_called_once()
    assert "Denied" in warning.call_args.args[-1]
    quit_app.assert_not_called()


@pytest.mark.parametrize("installed", [False, True])
def test_update_controls_visibility_busy_state_and_clicks(application, monkeypatch, installed):
    clicks = []
    monkeypatch.setattr(updater, "can_self_update", lambda: installed)
    monkeypatch.setattr(UpdateController, "update_now", lambda self: clicks.append("update"))
    monkeypatch.setattr(tray, "list_cameras", lambda count: [])
    monkeypatch.setattr(tray.devices, "list_devices", lambda kind: [])
    monkeypatch.setattr(tray.devices, "resolve", lambda kind, saved: None)
    manager = SimpleNamespace(
        config=PipelineConfig(check_for_updates=False),
        running=False, update_config=Mock(return_value=False), stop=Mock(),
    )
    instance = tray.SystemTrayApp(manager)
    try:
        assert instance.panel.update_now_btn.isHidden() is not installed
        assert instance.update_now_action.isVisible() is installed
        instance.updates._set_busy(True)
        assert not instance.panel.update_now_btn.isEnabled()
        assert not instance.update_now_action.isEnabled()
        instance.updates._set_busy(False)
        assert instance.panel.update_now_btn.isEnabled()
        assert instance.update_now_action.isEnabled() is installed
        if installed:
            instance.panel.update_now_btn.click()
            instance.update_now_action.trigger()
            assert clicks == ["update", "update"]
    finally:
        instance.updates._timer.stop()
        instance.panel.deleteLater()
        instance._menu.deleteLater()
        instance.deleteLater()
