"""Qt glue for update checks and the Windows self-updater."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from open_broadcast import __version__, updater

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineManager

log = logging.getLogger(__name__)

CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000  # every 6 hours while running
STARTUP_DELAY_MS = 5_000


class UpdateController(QObject):
    busy_changed = Signal(bool)
    _checked = Signal(object, object, bool, bool)
    _progress = Signal(int, int)
    _downloaded = Signal(object, object)  # path | None, error | None

    def __init__(self, manager: PipelineManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._busy = False
        self._progress_dialog: QProgressDialog | None = None
        self._download_cancelled: threading.Event | None = None
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._downloaded.connect(self._on_downloaded)

        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: self.check(interactive=False))

    def start_background_checks(self) -> None:
        if not self.manager.config.check_for_updates:
            return
        QTimer.singleShot(STARTUP_DELAY_MS, lambda: self.check(interactive=False))
        self._timer.start(CHECK_INTERVAL_MS)

    def set_enabled(self, enabled: bool) -> None:
        self.manager.update_config(check_for_updates=enabled)
        if enabled:
            self._timer.start(CHECK_INTERVAL_MS)
        else:
            self._timer.stop()

    # --- checking -----------------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        if self._busy != busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def update_now(self) -> None:
        self.check(interactive=True, install=True)

    def check(self, interactive: bool = True, *, install: bool = False) -> None:
        if self._busy or (not interactive and not self.manager.config.check_for_updates):
            return
        self._set_busy(True)

        def work():
            try:
                self._checked.emit(updater.check_for_update(), None, interactive, install)
            except Exception as e:
                log.warning("Update check failed: %s", e)
                self._checked.emit(None, str(e), interactive, install)

        threading.Thread(target=work, name="update-check", daemon=True).start()

    def _on_checked(self, release, error, interactive: bool, install: bool = False) -> None:
        self._set_busy(False)
        if not interactive and not self.manager.config.check_for_updates:
            return
        if error:
            if interactive:
                QMessageBox.warning(
                    None, "Open Broadcast", f"Could not check for updates:\n{error}"
                )
            return
        if release is None:
            if interactive:
                QMessageBox.information(
                    None, "Open Broadcast", f"You're up to date (v{__version__})."
                )
            return
        if not interactive and release.version == self.manager.config.skipped_version:
            return
        if interactive and install and updater.can_self_update():
            self._download_and_install(release)
        else:
            self._prompt(release)

    def _prompt(self, release: updater.ReleaseInfo) -> None:
        self_update = updater.can_self_update()
        box = QMessageBox()
        box.setWindowTitle("Update available")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Open Broadcast {release.version} is available (you have {__version__}).")
        if release.notes:
            box.setDetailedText(release.notes)
        install = box.addButton(
            "Install and restart" if self_update else "Open download page",
            QMessageBox.ButtonRole.AcceptRole,
        )
        skip = box.addButton("Skip this version", QMessageBox.ButtonRole.RejectRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked is skip:
            self.manager.update_config(skipped_version=release.version)
        elif clicked is install:
            if self_update:
                self._download_and_install(release)
            else:
                QDesktopServices.openUrl(QUrl(release.html_url))

    # --- installing (Windows installer builds) -----------------------------------------------

    def _download_and_install(self, release: updater.ReleaseInfo) -> None:
        if self._busy:
            return
        name = updater.platform_asset_name(release.version)
        asset = release.asset(name) if name else None
        if asset is None:
            QMessageBox.warning(None, "Open Broadcast",
                                "No installer found for this platform; opening the release page.")
            QDesktopServices.openUrl(QUrl(release.html_url))
            return

        self._set_busy(True)
        dlg = QProgressDialog(f"Downloading {asset.name}…", "Cancel", 0, 100)
        dlg.setWindowTitle("Updating Open Broadcast")
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        self._progress_dialog = dlg
        cancelled = threading.Event()
        self._download_cancelled = cancelled
        dlg.canceled.connect(cancelled.set)
        dlg.show()

        def progress(done: int, total: int) -> None:
            if cancelled.is_set():
                raise RuntimeError("Cancelled")
            self._progress.emit(done, total)

        def work():
            try:
                path = updater.download_asset(release, asset, progress=progress)
                if cancelled.is_set():
                    path.unlink(missing_ok=True)
                    raise RuntimeError("Cancelled")
                self._downloaded.emit(path, None)
            except Exception as e:
                self._downloaded.emit(None, str(e))

        threading.Thread(target=work, name="update-download", daemon=True).start()

    def _on_progress(self, done: int, total: int) -> None:
        if self._progress_dialog and total:
            self._progress_dialog.setValue(int(done * 100 / total))

    def _on_downloaded(self, path, error) -> None:
        self._set_busy(False)
        if self._download_cancelled is not None and self._download_cancelled.is_set():
            if path is not None:
                path.unlink(missing_ok=True)
            error = "Cancelled"
        self._download_cancelled = None
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        if error:
            if error != "Cancelled":
                QMessageBox.warning(None, "Open Broadcast", f"Update failed:\n{error}")
            return
        try:
            self.manager.stop()
            updater.launch_installer(path)
        except Exception as e:
            QMessageBox.warning(None, "Open Broadcast", f"Could not start installer:\n{e}")
            return
        QApplication.quit()
