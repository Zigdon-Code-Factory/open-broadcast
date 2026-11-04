"""Control panel window + system tray icon."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from open_broadcast import __version__, paths, updater
from open_broadcast.config import BACKGROUND_MODES
from open_broadcast.pipeline import devices
from open_broadcast.pipeline.video import list_cameras
from open_broadcast.ui.update import UpdateController

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineManager

log = logging.getLogger(__name__)

PREVIEW_SIZE = (352, 198)
MAX_CAMERAS = 5


def app_icon() -> QIcon:
    return QIcon(str(paths.assets_dir() / "icon.png"))


class ControlPanel(QWidget):
    """Main settings window."""

    _start_finished = Signal()

    def __init__(self, manager: PipelineManager, updates: UpdateController):
        super().__init__()
        self.manager = manager
        self.updates = updates
        self._starting = False
        self.setWindowTitle("Open Broadcast")
        self.setWindowIcon(app_icon())
        self.setMinimumWidth(400)
        self._build_ui()
        self._load_from_config()
        self._start_finished.connect(self._on_start_finished)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(66)

    # --- layout -------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        cfg = self.manager.config
        root = QVBoxLayout(self)

        # Video
        video = QGroupBox("Camera")
        video.setCheckable(True)
        video.toggled.connect(lambda v: self._set(video_enabled=v))
        self.video_group = video
        vf = QFormLayout(video)

        self.camera_combo = QComboBox()
        for i, name in list_cameras(MAX_CAMERAS):
            self.camera_combo.addItem(name, i)
        self.camera_combo.currentIndexChanged.connect(
            lambda _: self._set(camera_index=self.camera_combo.currentData())
        )
        vf.addRow("Device:", self.camera_combo)

        self.bg_combo = QComboBox()
        for mode in BACKGROUND_MODES:
            self.bg_combo.addItem(mode.capitalize(), mode)
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        vf.addRow("Background:", self.bg_combo)

        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(5, 99)
        self.blur_slider.valueChanged.connect(lambda v: self._set(blur_strength=v | 1))
        vf.addRow("Blur strength:", self.blur_slider)

        img_row = QHBoxLayout()
        self.bg_image_label = QLabel("(none)")
        self.bg_image_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pick = QPushButton("Choose…")
        pick.clicked.connect(self._pick_background)
        img_row.addWidget(self.bg_image_label, 1)
        img_row.addWidget(pick)
        vf.addRow("Image:", img_row)

        self.auto_frame_cb = QCheckBox("Auto frame (follow my face)")
        self.auto_frame_cb.toggled.connect(lambda v: self._set(auto_frame=v))
        vf.addRow(self.auto_frame_cb)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(100, 250)
        self.zoom_slider.valueChanged.connect(lambda v: self._set(auto_frame_zoom=v / 100))
        vf.addRow("Zoom:", self.zoom_slider)

        self.preview_cb = QCheckBox("Show preview")
        self.preview_cb.setChecked(True)
        vf.addRow(self.preview_cb)
        self.preview = QLabel()
        self.preview.setFixedSize(*PREVIEW_SIZE)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#888;border-radius:4px")
        self.preview.setText("Stopped")
        self.preview_cb.toggled.connect(self.preview.setVisible)
        vf.addRow(self.preview)
        root.addWidget(video)

        # Audio
        audio = QGroupBox("Microphone")
        audio.setCheckable(True)
        audio.toggled.connect(lambda v: self._set(audio_enabled=v))
        self.audio_group = audio
        af = QFormLayout(audio)

        self.input_combo = QComboBox()
        self.input_combo.currentIndexChanged.connect(
            lambda _: self._set(input_device=self.input_combo.currentData())
        )
        af.addRow("Input:", self.input_combo)

        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(
            lambda _: self._set(output_device=self.output_combo.currentData())
        )
        self.output_combo.setToolTip(
            "Send processed audio here. Pick a virtual cable (e.g. 'CABLE Input'), then select "
            "its matching microphone ('CABLE Output') in Discord/Zoom/etc."
        )
        af.addRow("Output:", self.output_combo)

        self.noise_cb = QCheckBox("Noise removal (DeepFilterNet3)")
        self.noise_cb.toggled.connect(lambda v: self._set(noise_suppression=v))
        af.addRow(self.noise_cb)

        self.vad_cb = QCheckBox("Mute when not speaking (Silero VAD)")
        self.vad_cb.toggled.connect(lambda v: self._set(vad_enabled=v))
        af.addRow(self.vad_cb)

        self.vad_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_slider.setRange(10, 90)
        self.vad_slider.valueChanged.connect(lambda v: self._set(vad_threshold=v / 100))
        af.addRow("VAD threshold:", self.vad_slider)

        meter_row = QHBoxLayout()
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(10)
        self.speech_label = QLabel("")
        self.speech_label.setFixedWidth(70)
        meter_row.addWidget(self.level, 1)
        meter_row.addWidget(self.speech_label)
        af.addRow("Level:", meter_row)

        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._populate_audio_devices)
        af.addRow(refresh)
        root.addWidget(audio)

        # Controls
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#c77")
        root.addWidget(self.status)

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self.toggle)
        root.addWidget(self.start_btn)

        self.update_now_btn = QPushButton("Update and restart")
        self.update_now_btn.setToolTip(
            "Check GitHub for a newer release, verify and install it in place, then restart. "
            "Your settings are kept."
        )
        self.update_now_btn.setVisible(updater.can_self_update())
        self.update_now_btn.clicked.connect(self.updates.update_now)
        self.updates.busy_changed.connect(
            lambda busy: self.update_now_btn.setEnabled(not busy)
        )
        root.addWidget(self.update_now_btn)

        footer = QHBoxLayout()
        footer.addWidget(QLabel(f"v{__version__}"))
        footer.addStretch(1)
        self.auto_update_cb = QCheckBox("Check for updates automatically")
        self.auto_update_cb.setChecked(cfg.check_for_updates)
        self.auto_update_cb.toggled.connect(self.updates.set_enabled)
        footer.addWidget(self.auto_update_cb)
        check = QPushButton("Check now")
        check.clicked.connect(lambda: self.updates.check(interactive=True))
        footer.addWidget(check)
        root.addLayout(footer)

    def _populate_audio_devices(self) -> None:
        cfg = self.manager.config
        for combo, kind, saved in (
            (self.input_combo, "input", cfg.input_device),
            (self.output_combo, "output", cfg.output_device),
        ):
            combo.blockSignals(True)
            combo.clear()
            devs = devices.list_devices(kind)
            for d in devs:
                combo.addItem(d.name, d.name)
            chosen = devices.resolve(kind, saved)
            if chosen is not None:
                combo.setCurrentIndex(max(0, combo.findData(chosen.name)))
            combo.blockSignals(False)
            if devs and saved is None and chosen is not None:
                # Persist the auto-picked default so it's stable across runs.
                self.manager.update_config(**{f"{kind}_device": chosen.name})

    def _load_from_config(self) -> None:
        cfg = self.manager.config
        widgets = [self.video_group, self.audio_group, self.camera_combo, self.bg_combo,
                   self.blur_slider, self.auto_frame_cb, self.zoom_slider, self.noise_cb,
                   self.vad_cb, self.vad_slider]
        for w in widgets:
            w.blockSignals(True)
        self.video_group.setChecked(cfg.video_enabled)
        self.audio_group.setChecked(cfg.audio_enabled)
        self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(cfg.camera_index)))
        self.bg_combo.setCurrentIndex(max(0, self.bg_combo.findData(cfg.background_mode)))
        self.blur_slider.setValue(cfg.blur_strength)
        self.auto_frame_cb.setChecked(cfg.auto_frame)
        self.zoom_slider.setValue(int(cfg.auto_frame_zoom * 100))
        self.noise_cb.setChecked(cfg.noise_suppression)
        self.vad_cb.setChecked(cfg.vad_enabled)
        self.vad_slider.setValue(int(cfg.vad_threshold * 100))
        if cfg.background_image:
            self.bg_image_label.setText(cfg.background_image.replace("\\", "/").rsplit("/", 1)[-1])
        for w in widgets:
            w.blockSignals(False)
        self._populate_audio_devices()
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        mode = self.bg_combo.currentData()
        self.blur_slider.setEnabled(mode == "blur")
        self.zoom_slider.setEnabled(self.auto_frame_cb.isChecked())
        self.vad_slider.setEnabled(self.vad_cb.isChecked())

    # --- actions ------------------------------------------------------------------------------

    def _set(self, **kwargs) -> None:
        needs_restart = self.manager.update_config(**kwargs)
        self._sync_enabled()
        if needs_restart and not self._starting:
            self.manager.stop()
            self._start()

    def _on_bg_changed(self, _index: int) -> None:
        mode = self.bg_combo.currentData()
        if mode == "replace" and not self.manager.config.background_image:
            self._pick_background()
        self._set(background_mode=mode)

    def _pick_background(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose background image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.bg_image_label.setText(path.rsplit("/", 1)[-1])
            self._set(background_image=path)

    def toggle(self) -> None:
        if self._starting:
            return
        if self.manager.running:
            self.manager.stop()
            self._refresh_state()
        else:
            self._start()

    def _start(self) -> None:
        self._starting = True
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Starting…")
        self.status.setText("")

        def work():
            try:
                self.manager.start()
            except Exception:
                log.exception("Failed to start pipelines")
            self._start_finished.emit()

        threading.Thread(target=work, name="pipeline-start", daemon=True).start()

    def _on_start_finished(self) -> None:
        self._starting = False
        self.start_btn.setEnabled(True)
        self._refresh_state()

    def _refresh_state(self) -> None:
        running = self.manager.running
        self.start_btn.setText("⏹ Stop" if running else "▶ Start")
        if not running:
            self.preview.clear()
            self.preview.setText("Stopped")
            self.level.setValue(0)
            self.speech_label.setText("")
            self.status.setText("")
        if callable(getattr(self, "on_state_changed", None)):
            self.on_state_changed(running)

    def _tick(self) -> None:
        if not self.manager.running:
            return
        errors = self.manager.errors()
        self.status.setText("\n".join(errors))

        video = self.manager.video
        if video is not None and self.preview_cb.isChecked() and self.isVisible():
            frame = video.latest_frame
            if frame is not None:
                small = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                self.preview.setPixmap(QPixmap.fromImage(img))
                self.preview.setToolTip(f"{video.fps:.0f} fps")

        audio = self.manager.audio
        if audio is not None:
            # Map RMS to a rough dBFS meter (-60..0 dB).
            import math

            db = 20 * math.log10(max(audio.level, 1e-6))
            self.level.setValue(int(max(0, min(100, (db + 60) / 60 * 100))))
            gated = self.manager.config.vad_enabled
            self.speech_label.setText(
                ("🎙 speaking" if audio.speech_active else "… muted") if gated else ""
            )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Closing the window keeps the app running in the tray.
        if QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
        else:
            self.manager.stop()
            event.accept()
            QApplication.quit()


class SystemTrayApp(QSystemTrayIcon):
    def __init__(self, manager: PipelineManager):
        super().__init__(app_icon())
        self.manager = manager
        self.updates = UpdateController(manager, self)
        self.panel = ControlPanel(manager, self.updates)
        self.panel.on_state_changed = self._on_state_changed

        self.setToolTip("Open Broadcast")
        self._build_menu()
        self.activated.connect(self._on_activated)

    def _build_menu(self) -> None:
        menu = QMenu()
        show_action = QAction("Show Controls", menu)
        show_action.triggered.connect(self.show_panel)
        menu.addAction(show_action)

        self.toggle_action = QAction("Start", menu)
        self.toggle_action.triggered.connect(self.panel.toggle)
        menu.addAction(self.toggle_action)

        menu.addSeparator()
        updates = QAction("Check for Updates…", menu)
        updates.triggered.connect(lambda: self.updates.check(interactive=True))
        menu.addAction(updates)

        self.update_now_action = QAction("Update and restart", menu)
        self.update_now_action.setVisible(updater.can_self_update())
        self.update_now_action.triggered.connect(self.updates.update_now)
        self.updates.busy_changed.connect(
            lambda busy: self.update_now_action.setEnabled(not busy)
        )
        menu.addAction(self.update_now_action)

        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)
        self._menu = menu
        self.setContextMenu(menu)

    def _on_state_changed(self, running: bool) -> None:
        self.toggle_action.setText("Stop" if running else "Start")
        self.setToolTip("Open Broadcast — " + ("running" if running else "stopped"))

    def show_panel(self) -> None:
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def _on_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_panel()

    def _quit(self) -> None:
        self.manager.stop()
        QApplication.quit()
