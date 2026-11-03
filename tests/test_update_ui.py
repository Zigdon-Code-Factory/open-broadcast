from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

from open_broadcast.ui.update import UpdateController


@pytest.fixture
def controller():
    application = QCoreApplication.instance() or QCoreApplication([])
    manager = SimpleNamespace(config=SimpleNamespace(check_for_updates=False))
    instance = UpdateController(manager)
    yield instance
    instance._timer.stop()
    assert application is not None


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
