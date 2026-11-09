"""Control panel window + system tray icon."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import cv2
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from open_broadcast import __version__, paths, updater
from open_broadcast.config import BACKGROUND_MODES, NOISE_REDUCTION_MAX_DB
from open_broadcast.pipeline import devices
from open_broadcast.pipeline.preview import VoicePreview
from open_broadcast.pipeline.video import list_cameras
from open_broadcast.ui.theme import STYLESHEET, tab_icon
from open_broadcast.ui.update import UpdateController

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineManager

log = logging.getLogger(__name__)

MAX_CAMERAS = 5
# Noise-removal strength notches (max attenuation in dB); the last one is unlimited.
NOISE_LEVELS_DB = (6.0, 10.0, 15.0, 20.0, 30.0, NOISE_REDUCTION_MAX_DB)
# Settings that change how the recorded voice preview sounds.
PREVIEW_KEYS = {"noise_suppression", "noise_reduction_db", "vad_enabled", "vad_threshold",
                "vad_holdover", "low_cut", "low_cut_hz"}


def noise_level_index(db: float) -> int:
    return min(range(len(NOISE_LEVELS_DB)), key=lambda i: abs(NOISE_LEVELS_DB[i] - db))


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def app_icon() -> QIcon:
    return QIcon(str(paths.assets_dir() / "icon.png"))


def label(text: str, role: str = "", *, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget


class ControlPanel(QWidget):
    """Main settings window."""

    _start_finished = Signal()
    _render_finished = Signal()

    def __init__(self, manager: PipelineManager, updates: UpdateController):
        super().__init__()
        self.manager = manager
        self.updates = updates
        self._starting = False
        self.voice_preview = VoicePreview(manager.config)
        self._rendering = False
        self._render_pending = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._render_preview)
        self.setWindowTitle("Open Broadcast")
        self.setWindowIcon(app_icon())
        self.setObjectName("controlPanel")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(820, 620)
        self.resize(1080, 760)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry().size()
            self.setMinimumSize(min(820, available.width()), min(620, available.height()))
            self.resize(min(1080, available.width()), min(760, available.height()))
        self._build_ui()
        self._load_from_config()
        self._start_finished.connect(self._on_start_finished)
        self._render_finished.connect(self._on_render_finished)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(66)
        self._refresh_state()

    # --- layout -------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        cfg = self.manager.config
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(18)

        header = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(36, 36))
        header.addWidget(logo)
        branding = QVBoxLayout()
        branding.setSpacing(2)
        branding.addWidget(label("OPEN BROADCAST", "brand"))
        branding.addWidget(label("Your devices. A cleaner signal.", "muted"))
        header.addLayout(branding)
        header.addStretch()
        self.pipeline_badge = label("STANDBY", "pipelineBadge")
        self.pipeline_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.pipeline_badge)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setIconSize(QSize(24, 24))
        self.tabs.setAccessibleName("Device categories")
        self.microphone_page = self._build_microphone_page()
        self.speaker_page = self._build_speaker_page()
        self.camera_page = self._build_camera_page()
        self.tabs.addTab(self.microphone_page, tab_icon("microphone"), "Microphone")
        self.tabs.addTab(self.speaker_page, tab_icon("speaker"), "Speaker")
        self.tabs.addTab(self.camera_page, tab_icon("camera"), "Camera")
        root.addWidget(self.tabs, 1)

        self.status = label("", "errorBox", wrap=True)
        self.status.hide()
        root.addWidget(self.status)

        self.settings_panel, settings = self._card("APPLICATION SETTINGS")
        self.settings_panel.hide()
        update_settings = QHBoxLayout()
        self.auto_update_cb = QCheckBox("Check for updates automatically")
        self.auto_update_cb.setChecked(cfg.check_for_updates)
        self.auto_update_cb.toggled.connect(self.updates.set_enabled)
        update_settings.addWidget(self.auto_update_cb)
        update_settings.addStretch()
        self.check_now_btn = QPushButton("Check now")
        self.check_now_btn.clicked.connect(lambda: self.updates.check(interactive=True))
        self.updates.busy_changed.connect(lambda busy: self.check_now_btn.setEnabled(not busy))
        update_settings.addWidget(self.check_now_btn)
        settings.addLayout(update_settings)
        root.addWidget(self.settings_panel)

        footer = QHBoxLayout()
        footer.setSpacing(12)
        info = QVBoxLayout()
        info.setSpacing(3)
        self.processing_hint = label("", "muted")
        info.addWidget(self.processing_hint)
        info.addWidget(label(f"v{__version__}  /  CPU processing", "helpText"))
        footer.addLayout(info)
        footer.addStretch()
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setCheckable(True)
        self.settings_btn.toggled.connect(self.settings_panel.setVisible)
        footer.addWidget(self.settings_btn)

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
        footer.addWidget(self.update_now_btn)
        self.start_btn = QPushButton("Start processing")
        self.start_btn.setObjectName("primary")
        self.start_btn.setMinimumWidth(170)
        self.start_btn.setToolTip(
            "Start or stop all enabled devices, regardless of the selected tab."
        )
        self.start_btn.clicked.connect(self.toggle)
        footer.addWidget(self.start_btn)
        root.addLayout(footer)

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(label(title, "eyebrow"))
        return card, layout

    def _page(self, title: str, subtitle: str) -> tuple[QScrollArea, QHBoxLayout]:
        content = QWidget()
        content.setObjectName("tabPage")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 20, 10, 4)
        layout.setSpacing(8)
        layout.addWidget(label(title, "pageTitle"))
        layout.addWidget(label(subtitle, "muted", wrap=True))
        layout.addSpacing(12)
        body = QHBoxLayout()
        body.setSpacing(20)
        layout.addLayout(body, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll, body

    def _controls(self, body: QHBoxLayout) -> QVBoxLayout:
        controls = QWidget()
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(370)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        body.addWidget(controls, 2)
        return layout

    def _build_microphone_page(self) -> QScrollArea:
        page, body = self._page("Microphone", "Make your voice the focus, not your room.")
        controls = self._controls(body)
        controls.setSpacing(12)
        self.audio_group = QGroupBox("Enable microphone")
        self.audio_group.setCheckable(True)
        self.audio_group.toggled.connect(lambda enabled: self._set(audio_enabled=enabled))
        sources = QVBoxLayout(self.audio_group)
        sources.setContentsMargins(16, 20, 16, 14)
        sources.setSpacing(8)
        sources.addWidget(label("INPUT DEVICE", "eyebrow"))
        self.input_combo = QComboBox()
        self.input_combo.setAccessibleName("Microphone input device")
        self.input_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.input_combo.setMinimumContentsLength(16)
        self.input_combo.setPlaceholderText("No microphone found")
        self.input_combo.currentIndexChanged.connect(
            lambda _: self._set(input_device=self.input_combo.currentData())
        )
        sources.addWidget(self.input_combo)
        sources.addWidget(label("PROCESSED OUTPUT", "eyebrow"))
        self.output_combo = QComboBox()
        self.output_combo.setAccessibleName("Processed microphone output device")
        self.output_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.output_combo.setMinimumContentsLength(16)
        self.output_combo.setPlaceholderText("No output device found")
        self.output_combo.setToolTip(
            "Choose CABLE Input here, then CABLE Output as the microphone in your calling app."
        )
        self.output_combo.currentIndexChanged.connect(
            lambda _: self._set(output_device=self.output_combo.currentData())
        )
        sources.addWidget(self.output_combo)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._populate_audio_devices)
        sources.addWidget(refresh)
        controls.addWidget(self.audio_group)

        self.audio_effects, effects = self._card("VOICE EFFECTS")
        effects.setSpacing(8)
        self.noise_cb = QCheckBox("Noise removal")
        self.noise_cb.setToolTip("DeepFilterNet3 removes fans, keyboards and room noise.")
        self.noise_cb.toggled.connect(lambda enabled: self._set(noise_suppression=enabled))
        effects.addWidget(self.noise_cb)
        self.noise_slider, self.noise_value = self._tuning_row(
            effects, "Strength", 0, len(NOISE_LEVELS_DB) - 1,
            "How far background noise is pushed down. Lower settings keep a little room "
            "tone and sound more natural; Max removes as much as possible.",
        )
        self.noise_slider.setPageStep(1)
        self.noise_slider.valueChanged.connect(
            lambda index: self._set(noise_reduction_db=NOISE_LEVELS_DB[index])
        )
        self.vad_cb = QCheckBox("Mute when not speaking")
        self.vad_cb.setToolTip("Silero VAD keeps the gaps between words silent.")
        self.vad_cb.toggled.connect(lambda enabled: self._set(vad_enabled=enabled))
        effects.addWidget(self.vad_cb)
        self.vad_slider, self.vad_value = self._tuning_row(
            effects, "Threshold", 10, 90,
            "Lower values let quieter speech through; higher values mute more aggressively.",
        )
        self.vad_slider.setAccessibleName("Voice detection threshold")
        self.vad_slider.valueChanged.connect(lambda value: self._set(vad_threshold=value / 100))
        self.hold_slider, self.hold_value = self._tuning_row(
            effects, "Hold", 1, 15,
            "How long the mic stays open after you stop talking, so word endings "
            "are not cut off.",
        )
        self.hold_slider.setPageStep(1)
        self.hold_slider.valueChanged.connect(lambda value: self._set(vad_holdover=value / 10))
        self.low_cut_cb = QCheckBox("Reduce low rumble")
        self.low_cut_cb.setToolTip(
            "80 Hz high-pass filter for desk thumps, HVAC hum and mic-stand bumps."
        )
        self.low_cut_cb.toggled.connect(lambda enabled: self._set(low_cut=enabled))
        effects.addWidget(self.low_cut_cb)
        controls.addWidget(self.audio_effects)
        controls.addStretch()

        monitor, monitor_layout = self._card("MICROPHONE MONITOR")
        # Leave room for native font/control metrics on macOS and Linux as well.
        monitor_layout.setSpacing(8)
        state_row = QHBoxLayout()
        self.microphone_state = label("Ready when you are", "sectionTitle")
        state_row.addWidget(self.microphone_state)
        state_row.addStretch()
        self.speech_label = label("", "accent")
        state_row.addWidget(self.speech_label)
        monitor_layout.addLayout(state_row)
        self.level = QProgressBar()
        self.level.setAccessibleName("Microphone input level")
        self.level.setRange(0, 100)
        self.level.setValue(0)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(22)
        monitor_layout.addWidget(self.level)
        scale = QHBoxLayout()
        scale.addWidget(label("-60 dB", "helpText"))
        scale.addStretch()
        scale.addWidget(label("0 dB", "helpText"))
        monitor_layout.addLayout(scale)
        monitor_layout.addStretch()
        self._build_voice_preview(monitor_layout)
        monitor_layout.addStretch()
        monitor_layout.addWidget(label("YOUR SIGNAL PATH", "eyebrow"))
        self.route_input = label("", wrap=True)
        self.route_output = label("", wrap=True)
        monitor_layout.addWidget(self.route_input)
        self.route_effects = label("", "accent")
        monitor_layout.addWidget(self.route_effects)
        monitor_layout.addWidget(self.route_output)
        self.route_hint = label("", "helpText", wrap=True)
        monitor_layout.addWidget(self.route_hint)
        body.addWidget(monitor, 3)
        return page

    def _tuning_row(self, parent: QVBoxLayout, name: str, low: int, high: int,
                    tooltip: str) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        row.setContentsMargins(26, 0, 0, 0)
        row.setSpacing(10)
        caption = label(name, "helpText")
        caption.setFixedWidth(64)
        caption.setToolTip(tooltip)
        row.addWidget(caption)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setAccessibleName(name)
        slider.setToolTip(tooltip)
        row.addWidget(slider, 1)
        value = label("", "helpText")
        value.setFixedWidth(44)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        parent.addLayout(row)
        return slider, value

    def _build_voice_preview(self, parent: QVBoxLayout) -> None:
        parent.addWidget(label("TEST YOUR VOICE", "eyebrow"))
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.record_btn = QPushButton("Record")
        self.record_btn.setToolTip(
            f"Record up to {VoicePreview.MAX_SECONDS:.0f} seconds from your microphone."
        )
        self.record_btn.clicked.connect(self._toggle_recording)
        buttons.addWidget(self.record_btn)
        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Plays on your default speakers or headphones.")
        self.play_btn.clicked.connect(self._toggle_playback)
        buttons.addWidget(self.play_btn)
        buttons.addStretch()
        self.source_group = QButtonGroup(self)
        self.source_group.setExclusive(True)
        self.source_buttons = {}
        for source, text in (("original", "Original"), ("processed", "Processed")):
            button = QPushButton(text)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setToolTip(
                "Hear the raw recording." if source == "original" else
                "Hear it with the current voice effects."
            )
            button.toggled.connect(
                lambda checked, source=source: checked and self._set_preview_source(source)
            )
            self.source_group.addButton(button)
            self.source_buttons[source] = button
            buttons.addWidget(button)
        parent.addLayout(buttons)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setAccessibleName("Playback position")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(lambda ms: self.voice_preview.seek(ms / 1000))
        self.seek_slider.sliderReleased.connect(
            lambda: self.voice_preview.seek(self.seek_slider.value() / 1000)
        )
        bar.addWidget(self.seek_slider, 1)
        self.preview_time = label("0:00 / 0:00", "helpText")
        bar.addWidget(self.preview_time)
        parent.addLayout(bar)
        self.preview_hint = label("", "helpText", wrap=True)
        parent.addWidget(self.preview_hint)
        self.source_buttons[self.voice_preview.source].setChecked(True)
        self._sync_preview()

    def _build_speaker_page(self) -> QScrollArea:
        page, body = self._page("Speaker", "A dedicated space for incoming audio.")
        card, content = self._card("SPEAKER PROCESSING")
        content.addStretch()
        content.addWidget(label("Not available yet", "pageTitle"))
        content.addWidget(label(
            "Incoming-audio noise removal is not implemented in this version. "
            "This tab does not process or change your speaker output.",
            "muted", wrap=True,
        ))
        content.addWidget(label(
            "Your microphone and camera controls are ready in their own tabs.",
            "helpText", wrap=True,
        ))
        self.speaker_back_btn = QPushButton("Go to Microphone")
        self.speaker_back_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        content.addWidget(self.speaker_back_btn, 0, Qt.AlignmentFlag.AlignLeft)
        content.addStretch()
        body.addWidget(card)
        return page

    def _build_camera_page(self) -> QScrollArea:
        page, body = self._page("Camera", "Keep the attention on you.")
        controls = self._controls(body)
        self.video_group = QGroupBox("Enable camera")
        self.video_group.setCheckable(True)
        self.video_group.toggled.connect(lambda enabled: self._set(video_enabled=enabled))
        sources = QVBoxLayout(self.video_group)
        sources.setContentsMargins(16, 20, 16, 14)
        sources.setSpacing(8)
        sources.addWidget(label("CAMERA SOURCE", "eyebrow"))
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Camera device")
        self.camera_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.camera_combo.setMinimumContentsLength(16)
        self.camera_combo.setPlaceholderText("No camera found")
        for camera_index, name in list_cameras(MAX_CAMERAS):
            self.camera_combo.addItem(name, camera_index)
        self.camera_combo.currentIndexChanged.connect(
            lambda _: self._set(camera_index=self.camera_combo.currentData())
        )
        sources.addWidget(self.camera_combo)
        controls.addWidget(self.video_group)

        self.camera_effects, effects = self._card("CAMERA EFFECTS")
        effects.addWidget(label("Background", "muted"))
        self.bg_combo = QComboBox()
        self.bg_combo.setAccessibleName("Background effect")
        for mode in BACKGROUND_MODES:
            self.bg_combo.addItem(mode.capitalize(), mode)
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        effects.addWidget(self.bg_combo)
        effects.addWidget(label("Blur strength", "helpText"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setAccessibleName("Background blur strength")
        self.blur_slider.setRange(5, 99)
        self.blur_slider.valueChanged.connect(lambda value: self._set(blur_strength=value | 1))
        effects.addWidget(self.blur_slider)
        self.background_row = QWidget()
        image_layout = QVBoxLayout(self.background_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.background_btn = QPushButton("Choose image")
        self.background_btn.clicked.connect(self._pick_background)
        image_layout.addWidget(self.background_btn)
        self.bg_image_label = label("(none)", "helpText", wrap=True)
        self.bg_image_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        image_layout.addWidget(self.bg_image_label)
        effects.addWidget(self.background_row)
        self.auto_frame_cb = QCheckBox("Auto frame (follow my face)")
        self.auto_frame_cb.toggled.connect(lambda enabled: self._set(auto_frame=enabled))
        effects.addWidget(self.auto_frame_cb)
        effects.addWidget(label("Framing zoom", "helpText"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setAccessibleName("Auto framing zoom")
        self.zoom_slider.setRange(100, 250)
        self.zoom_slider.valueChanged.connect(lambda value: self._set(auto_frame_zoom=value / 100))
        effects.addWidget(self.zoom_slider)
        controls.addWidget(self.camera_effects)
        controls.addStretch()

        preview_card, preview_layout = self._card("CAMERA PREVIEW")
        self.preview_cb = QCheckBox("Show preview")
        self.preview_cb.setChecked(True)
        preview_layout.addWidget(self.preview_cb)
        self.preview = label("Camera preview\n\nStart processing to see your camera.", "preview",
                             wrap=True)
        self.preview.setAccessibleName("Camera preview")
        self.preview.setMinimumSize(300, 200)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_cb.toggled.connect(self.preview.setVisible)
        preview_layout.addWidget(self.preview, 1)
        self.preview_info = label("Live preview stays on your device.", "helpText", wrap=True)
        preview_layout.addWidget(self.preview_info)
        preview_layout.addWidget(label(
            "Calling apps need a compatible virtual-camera driver.",
            "helpText", wrap=True,
        ))
        body.addWidget(preview_card, 3)
        return page

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
        self._sync_enabled()

    def _load_from_config(self) -> None:
        cfg = self.manager.config
        widgets = [self.video_group, self.audio_group, self.camera_combo, self.bg_combo,
                   self.blur_slider, self.auto_frame_cb, self.zoom_slider, self.noise_cb,
                   self.vad_cb, self.vad_slider, self.noise_slider, self.hold_slider,
                   self.low_cut_cb]
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
        self.vad_slider.setValue(round(cfg.vad_threshold * 100))
        self.noise_slider.setValue(noise_level_index(cfg.noise_reduction_db))
        self.hold_slider.setValue(round(cfg.vad_holdover * 10))
        self.low_cut_cb.setChecked(cfg.low_cut)
        if cfg.background_image:
            self.bg_image_label.setText(cfg.background_image.replace("\\", "/").rsplit("/", 1)[-1])
        for w in widgets:
            w.blockSignals(False)
        self._populate_audio_devices()
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        mode = self.bg_combo.currentData()
        self.blur_slider.setEnabled(mode == "blur")
        self.background_row.setVisible(mode == "replace")
        self.zoom_slider.setEnabled(self.auto_frame_cb.isChecked())
        self.noise_slider.setEnabled(self.noise_cb.isChecked())
        self.vad_slider.setEnabled(self.vad_cb.isChecked())
        self.hold_slider.setEnabled(self.vad_cb.isChecked())
        noise_db = NOISE_LEVELS_DB[self.noise_slider.value()]
        self.noise_value.setText(
            "Max" if noise_db >= NOISE_REDUCTION_MAX_DB else f"{noise_db:.0f} dB"
        )
        self.vad_value.setText(f"{self.vad_slider.value()}%")
        self.hold_value.setText(f"{self.hold_slider.value() / 10:.1f} s")
        self.audio_effects.setEnabled(self.audio_group.isChecked())
        self.camera_effects.setEnabled(self.video_group.isChecked())
        enabled = []
        if self.audio_group.isChecked():
            enabled.append("Microphone")
        if self.video_group.isChecked():
            enabled.append("Camera")
        self.processing_hint.setText(" + ".join(enabled) or "No devices enabled")
        self.start_btn.setEnabled(
            not self._starting and (self.manager.running or bool(enabled))
        )
        self.route_input.setText(self.input_combo.currentText() or "No microphone found")
        output = self.output_combo.currentText()
        self.route_output.setText(output or "No output device found")
        effects = []
        if self.noise_cb.isChecked():
            effects.append("Noise removal")
        if self.vad_cb.isChecked():
            effects.append("Voice gate")
        if self.low_cut_cb.isChecked():
            effects.append("Low cut")
        self.route_effects.setText(" + ".join(effects) or "Passthrough / effects off")
        self.route_hint.setText(
            "In Discord or Zoom, choose CABLE Output as your microphone."
            if "cable input" in output.lower() else
            "Choose a virtual output here, then its matching microphone in your calling app."
        )

    # --- actions ------------------------------------------------------------------------------

    def _set(self, **kwargs) -> None:
        needs_restart = self.manager.update_config(**kwargs)
        self._sync_enabled()
        if PREVIEW_KEYS & kwargs.keys() and self.voice_preview.has_clip:
            self._render_timer.start()
        if needs_restart and not self._starting:
            self.manager.stop()
            if self.manager.config.audio_enabled or self.manager.config.video_enabled:
                self._start()
            else:
                self._refresh_state()

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

    def _toggle_recording(self) -> None:
        preview = self.voice_preview
        if preview.recording:
            preview.stop_recording()
            if preview.has_clip:
                self._render_preview()
        else:
            preview.start_recording()
        self._sync_preview()

    def _toggle_playback(self) -> None:
        preview = self.voice_preview
        if preview.playing:
            preview.pause()
        else:
            preview.play()
        self._sync_preview()

    def _set_preview_source(self, source: str) -> None:
        self.voice_preview.set_source(source)
        self._sync_preview()

    def _render_preview(self) -> None:
        if self._rendering:
            self._render_pending = True
            return
        self._rendering = True
        self._render_pending = False
        self._sync_preview()

        def work():
            try:
                self.voice_preview.render()
            except Exception:
                log.exception("Failed to render voice preview")
            self._render_finished.emit()

        threading.Thread(target=work, name="preview-render", daemon=True).start()

    def _on_render_finished(self) -> None:
        self._rendering = False
        if self._render_pending:
            self._render_preview()
        self._sync_preview()

    def _sync_preview(self) -> None:
        preview = self.voice_preview
        if preview.recording_finished:
            self._toggle_recording()
            return
        recording = preview.recording
        waiting = preview.source == "processed" and preview.processed is None
        self.record_btn.setText("Stop" if recording else "Record")
        self.play_btn.setText("Pause" if preview.playing else "Play")
        self.play_btn.setEnabled(preview.has_clip and not recording
                                 and (preview.playing or not waiting))
        self.seek_slider.setEnabled(preview.has_clip and not recording)
        total_ms = int(preview.duration * 1000)
        if self.seek_slider.maximum() != total_ms:
            self.seek_slider.setRange(0, total_ms)
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(0 if recording else int(preview.elapsed * 1000))
        total = preview.MAX_SECONDS if recording else preview.duration
        self.preview_time.setText(f"{format_time(preview.elapsed)} / {format_time(total)}")
        if preview.error:
            hint = preview.error
        elif recording:
            hint = "Recording... speak normally, with your usual background noise."
        elif self._rendering and preview.has_clip:
            hint = "Applying effects..."
        elif preview.has_clip:
            hint = "Switch Original / Processed while playing; effect changes apply to the clip."
        else:
            hint = f"Record a short clip (up to {preview.MAX_SECONDS:.0f} s), then compare."
        self.preview_hint.setText(hint)

    def toggle(self) -> None:
        if self._starting:
            return
        if self.manager.running:
            self.manager.stop()
            self._refresh_state()
        else:
            self._start()

    def _start(self) -> None:
        if not (self.manager.config.audio_enabled or self.manager.config.video_enabled):
            return
        self._starting = True
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Starting…")
        self.status.setText("")
        self.status.hide()
        self.pipeline_badge.setText("STARTING")

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
        self.start_btn.setText("Stop processing" if running else "Start processing")
        self.pipeline_badge.setText("PROCESSING" if running else "STANDBY")
        self.pipeline_badge.setProperty("running", running)
        self.pipeline_badge.style().unpolish(self.pipeline_badge)
        self.pipeline_badge.style().polish(self.pipeline_badge)
        self.microphone_state.setText(
            "Live microphone" if running and self.manager.config.audio_enabled else
            "Ready when you are" if self.manager.config.audio_enabled else "Microphone disabled"
        )
        self.start_btn.setEnabled(
            not self._starting and (
                running or self.manager.config.audio_enabled or self.manager.config.video_enabled
            )
        )
        if not running:
            self.preview.clear()
            self.preview.setText("Camera preview\n\nStart processing to see your camera.")
            self.preview_info.setText("Live preview stays on your device.")
            self.level.setValue(0)
            self.speech_label.setText("")
            self.status.setText("")
            self.status.hide()
        if callable(getattr(self, "on_state_changed", None)):
            self.on_state_changed(running)

    def _tick(self) -> None:
        if self.isVisible():
            self._sync_preview()
        if not self.manager.running:
            return
        errors = self.manager.errors()
        self.status.setText("\n".join(errors))
        self.status.setVisible(bool(errors))

        video = self.manager.video
        if (video is not None and self.preview_cb.isChecked() and self.isVisible()
                and self.tabs.currentWidget() is self.camera_page):
            frame = video.latest_frame
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width = rgb.shape[:2]
                image = QImage(
                    rgb.data, frame_width, frame_height, 3 * frame_width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.preview.setPixmap(QPixmap.fromImage(image).scaled(
                    self.preview.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
                self.preview_info.setText(f"Live preview  /  {video.fps:.0f} fps")

        audio = self.manager.audio
        if audio is not None:
            # Map RMS to a rough dBFS meter (-60..0 dB).
            import math

            db = 20 * math.log10(max(audio.level, 1e-6))
            self.level.setValue(int(max(0, min(100, (db + 60) / 60 * 100))))
            gated = self.manager.config.vad_enabled
            self.speech_label.setText(
                ("Speech detected" if audio.speech_active else "Voice gate / muted")
                if gated else "Voice gate off"
            )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Closing the window keeps the app running in the tray.
        self.voice_preview.pause()
        if QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
        else:
            self.voice_preview.close()
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
        self.panel.voice_preview.close()
        self.manager.stop()
        QApplication.quit()
