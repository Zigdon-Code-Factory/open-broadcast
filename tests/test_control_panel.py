from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QLabel

from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline.devices import AudioDevice
from open_broadcast.ui import tray
from open_broadcast.ui.update import UpdateController


@pytest.fixture
def panel(application, monkeypatch):
    inputs = [AudioDevice(0, "Antlion Wireless Microphone", 1, 0)]
    outputs = [
        AudioDevice(1, "CABLE Input (VB-Audio Virtual Cable)", 0, 2),
        AudioDevice(2, "Headphones", 0, 2),
    ]
    monkeypatch.setattr(tray, "list_cameras", lambda count: [(0, "Webcam"), (1, "USB Camera")])
    monkeypatch.setattr(
        tray.devices, "list_devices", lambda kind: inputs if kind == "input" else outputs
    )
    monkeypatch.setattr(
        tray.devices, "resolve", lambda kind, saved: inputs[0] if kind == "input" else outputs[0]
    )
    config = PipelineConfig(
        input_device=inputs[0].name, output_device=outputs[0].name, check_for_updates=False
    )
    manager = SimpleNamespace(
        config=config, running=False, audio=None, video=None,
        start=Mock(), stop=Mock(), errors=Mock(return_value=[]),
    )

    def update_config(**changes):
        for key, value in changes.items():
            setattr(config, key, value)
        return False

    manager.update_config = Mock(side_effect=update_config)
    updates = UpdateController(manager)
    instance = tray.ControlPanel(manager, updates)
    instance._timer.stop()
    yield instance
    instance.hide()
    updates._timer.stop()
    instance.deleteLater()
    updates.deleteLater()
    application.processEvents()


def test_device_tabs_and_speaker_unavailability(panel):
    assert [panel.tabs.tabText(index) for index in range(panel.tabs.count())] == [
        "Microphone", "Speaker", "Camera"
    ]
    assert panel.tabs.currentWidget() is panel.microphone_page
    assert panel.microphone_page.isAncestorOf(panel.input_combo)
    assert panel.camera_page.isAncestorOf(panel.camera_combo)
    assert panel.speaker_page.findChildren(QComboBox) == []
    assert any(
        widget.text() == "Not available yet"
        for widget in panel.speaker_page.findChildren(QLabel)
    )


def test_switching_tabs_preserves_config_and_does_not_restart(panel):
    for index in (2, 1, 0, 2):
        panel.tabs.setCurrentIndex(index)
    panel.manager.update_config.assert_not_called()
    panel.manager.start.assert_not_called()
    panel.manager.stop.assert_not_called()
    assert panel.manager.config.audio_enabled
    assert panel.manager.config.video_enabled


def test_speaker_navigation_returns_to_microphone(panel):
    panel.tabs.setCurrentWidget(panel.speaker_page)
    panel.speaker_back_btn.click()
    assert panel.tabs.currentWidget() is panel.microphone_page
    panel.manager.update_config.assert_not_called()


@pytest.mark.parametrize("widget,method,value,key,expected", [
    ("input_combo", "setCurrentIndex", -1, "input_device", None),
    ("output_combo", "setCurrentIndex", 1, "output_device", "Headphones"),
    ("camera_combo", "setCurrentIndex", 1, "camera_index", 1),
    ("bg_combo", "setCurrentIndex", 3, "background_mode", "off"),
    ("blur_slider", "setValue", 40, "blur_strength", 41),
    ("auto_frame_cb", "setChecked", True, "auto_frame", True),
    ("zoom_slider", "setValue", 180, "auto_frame_zoom", 1.8),
    ("noise_cb", "setChecked", False, "noise_suppression", False),
    ("vad_cb", "setChecked", False, "vad_enabled", False),
    ("vad_slider", "setValue", 35, "vad_threshold", 0.35),
    ("noise_slider", "setValue", 3, "noise_reduction_db", 20.0),
    ("hold_slider", "setValue", 8, "vad_holdover", 0.8),
    ("low_cut_cb", "setChecked", True, "low_cut", True),
    ("audio_group", "setChecked", False, "audio_enabled", False),
    ("video_group", "setChecked", False, "video_enabled", False),
])
def test_effect_controls_still_update_pipeline_config(panel, widget, method, value, key, expected):
    getattr(getattr(panel, widget), method)(value)
    assert getattr(panel.manager.config, key) == expected
    panel.manager.update_config.assert_called_with(**{key: expected})


def test_effect_availability_tracks_toggles(panel):
    panel.vad_cb.setChecked(False)
    assert not panel.vad_slider.isEnabled()
    assert not panel.zoom_slider.isEnabled()
    panel.auto_frame_cb.setChecked(True)
    assert panel.zoom_slider.isEnabled()
    panel.audio_group.setChecked(False)
    assert not panel.audio_effects.isEnabled()
    panel.video_group.setChecked(False)
    assert not panel.camera_effects.isEnabled()
    assert panel.processing_hint.text() == "No devices enabled"
    assert not panel.start_btn.isEnabled()
    panel._start()
    panel.manager.start.assert_not_called()
    panel.audio_group.setChecked(True)
    assert panel.audio_effects.isEnabled()
    assert panel.start_btn.isEnabled()


def test_existing_config_loads_without_writing_or_starting(panel):
    config = panel.manager.config
    config.video_enabled = False
    config.vad_enabled = False
    config.vad_threshold = 0.35
    config.background_mode = "replace"
    config.background_image = "C:/Pictures/room.png"
    config.auto_frame = True
    config.auto_frame_zoom = 1.8
    panel._load_from_config()
    assert not panel.video_group.isChecked()
    assert not panel.vad_cb.isChecked()
    assert panel.vad_slider.value() == 35
    assert not panel.vad_slider.isEnabled()
    assert panel.bg_combo.currentData() == "replace"
    assert not panel.background_row.isHidden()
    assert panel.bg_image_label.text() == "room.png"
    assert panel.auto_frame_cb.isChecked()
    assert panel.zoom_slider.value() == 180
    panel.manager.update_config.assert_not_called()
    panel.manager.start.assert_not_called()


def test_microphone_routing_explains_virtual_cable(panel):
    assert panel.route_input.text() == "Antlion Wireless Microphone"
    assert "CABLE Input" in panel.route_output.text()
    assert "CABLE Output" in panel.route_hint.text()
    panel.output_combo.setCurrentIndex(1)
    assert panel.route_output.text() == "Headphones"
    assert "virtual output" in panel.route_hint.text()
    panel.noise_cb.setChecked(False)
    panel.vad_cb.setChecked(False)
    assert "effects off" in panel.route_effects.text()


def test_settings_preserves_update_checks_and_busy_state(panel, monkeypatch):
    check = Mock()
    monkeypatch.setattr(panel.updates, "check", check)
    assert panel.settings_panel.isHidden()
    panel.settings_btn.click()
    assert not panel.settings_panel.isHidden()
    panel.check_now_btn.click()
    check.assert_called_once_with(interactive=True)
    panel.auto_update_cb.setChecked(True)
    assert panel.manager.config.check_for_updates
    panel.updates._set_busy(True)
    assert not panel.check_now_btn.isEnabled()
    panel.updates._set_busy(False)
    assert panel.check_now_btn.isEnabled()
    panel.settings_btn.click()
    assert panel.settings_panel.isHidden()


def test_running_meter_gate_and_stop_reset(panel):
    panel.manager.running = True
    panel.manager.audio = SimpleNamespace(level=0.1, speech_active=False)
    panel._refresh_state()
    panel.tabs.setCurrentWidget(panel.speaker_page)
    panel._tick()
    assert panel.pipeline_badge.text() == "PROCESSING"
    assert panel.microphone_state.text() == "Live microphone"
    assert 65 <= panel.level.value() <= 67
    assert "muted" in panel.speech_label.text()
    panel.manager.config.vad_enabled = False
    panel._tick()
    assert panel.speech_label.text() == "Voice gate off"
    panel.manager.running = False
    panel._refresh_state()
    assert panel.pipeline_badge.text() == "STANDBY"
    assert panel.level.value() == 0
    assert panel.speech_label.text() == ""


def test_camera_preview_only_renders_on_camera_tab(panel, application, monkeypatch):
    panel.manager.running = True
    panel.manager.video = SimpleNamespace(
        latest_frame=np.zeros((360, 640, 3), dtype=np.uint8), fps=30
    )
    convert = Mock(wraps=tray.cv2.cvtColor)
    monkeypatch.setattr(tray.cv2, "cvtColor", convert)
    panel.show()
    application.processEvents()
    panel._tick()
    convert.assert_not_called()
    panel.tabs.setCurrentWidget(panel.camera_page)
    application.processEvents()
    panel._tick()
    convert.assert_called_once()
    preview = panel.preview.pixmap()
    assert not preview.isNull()
    assert preview.width() <= panel.preview.width()
    assert preview.height() <= panel.preview.height()
    assert preview.width() / preview.height() == pytest.approx(16 / 9, abs=0.02)
    assert "30 fps" in panel.preview_info.text()
    panel.preview_cb.setChecked(False)
    panel._tick()
    convert.assert_called_once()


def test_errors_visible_on_every_tab(panel, application):
    panel.show()
    panel.manager.running = True
    panel.manager.errors.return_value = ["Audio device unavailable"]
    for index in range(panel.tabs.count()):
        panel.tabs.setCurrentIndex(index)
        application.processEvents()
        panel._tick()
        assert panel.status.isVisible()
        assert panel.status.text() == "Audio device unavailable"
    panel.manager.errors.return_value = []
    panel._tick()
    assert panel.status.isHidden()


def test_compact_window_keeps_tabs_and_global_controls_accessible(panel, application):
    panel.resize(840, 620)
    panel.show()
    panel.settings_btn.click()
    for index in range(panel.tabs.count()):
        panel.tabs.setCurrentIndex(index)
        application.processEvents()
        page = panel.tabs.currentWidget()
        assert page.horizontalScrollBar().maximum() == 0
        assert panel.start_btn.isVisible()
        assert panel.rect().contains(panel.start_btn.geometry())
        assert panel.check_now_btn.isVisible()
        assert panel.rect().contains(panel.settings_panel.geometry())


def test_default_window_fits_all_tabs_without_scrolling(panel, application):
    panel.resize(1080, 760)
    panel.show()
    for index in range(panel.tabs.count()):
        panel.tabs.setCurrentIndex(index)
        application.processEvents()
        page = panel.tabs.currentWidget()
        assert page.horizontalScrollBar().maximum() == 0
        assert page.verticalScrollBar().maximum() == 0, panel.tabs.tabText(index)


def test_start_stop_controls_and_loading_state(panel, monkeypatch):
    targets = []
    monkeypatch.setattr(
        tray.threading, "Thread",
        lambda target, **kwargs: SimpleNamespace(start=lambda: targets.append(target)),
    )
    panel.manager.start.side_effect = lambda: setattr(panel.manager, "running", True)
    panel.manager.stop.side_effect = lambda: setattr(panel.manager, "running", False)
    panel.start_btn.click()
    assert not panel.start_btn.isEnabled()
    assert panel.pipeline_badge.text() == "STARTING"
    panel.toggle()
    assert len(targets) == 1
    targets[0]()
    assert panel.start_btn.isEnabled()
    assert panel.pipeline_badge.text() == "PROCESSING"
    assert panel.start_btn.text() == "Stop processing"
    panel.start_btn.click()
    assert panel.pipeline_badge.text() == "STANDBY"
    panel.manager.start.assert_called_once()
    panel.manager.stop.assert_called_once()


def test_tuning_controls_load_and_label_values(panel):
    config = panel.manager.config
    config.noise_reduction_db = 14.0
    config.vad_holdover = 0.5
    config.low_cut = True
    panel._load_from_config()
    assert panel.noise_slider.value() == tray.NOISE_LEVELS_DB.index(15.0)
    assert panel.noise_value.text() == "15 dB"
    assert panel.hold_value.text() == "0.5 s"
    assert panel.low_cut_cb.isChecked()
    assert "Low cut" in panel.route_effects.text()
    panel.noise_slider.setValue(panel.noise_slider.maximum())
    assert panel.noise_value.text() == "Max"
    panel.noise_cb.setChecked(False)
    assert not panel.noise_slider.isEnabled()
    panel.vad_cb.setChecked(False)
    assert not panel.hold_slider.isEnabled()
    panel.manager.update_config.reset_mock()
    panel._load_from_config()
    panel.manager.update_config.assert_not_called()


class FakePreview:
    MAX_SECONDS = 10.0

    def __init__(self):
        self.source = "processed"
        self.recording = self.playing = False
        self.hit_limit = False
        self.original = self.processed = None
        self.position = 0
        self.error = None
        self.renders = 0
        self.calls = []

    recording_finished = property(lambda self: self.recording and self.hit_limit)
    has_clip = property(lambda self: self.original is not None)
    duration = property(lambda self: 0 if self.original is None else len(self.original) / 48000)
    elapsed = property(lambda self: self.position / 48000)

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.original = np.zeros(48000 * 3, np.float32)

    def render(self):
        self.renders += 1
        self.processed = self.original

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def seek(self, seconds):
        self.position = int(seconds * 48000)

    def set_source(self, source):
        self.source = source

    def close(self):
        pass


def test_voice_preview_record_play_and_rerender(panel, monkeypatch):
    threads = []
    monkeypatch.setattr(
        tray.threading, "Thread",
        lambda target, **kwargs: SimpleNamespace(start=lambda: threads.append(target)),
    )
    fake = FakePreview()
    panel.voice_preview = fake
    panel._sync_preview()
    assert not panel.play_btn.isEnabled()
    assert panel.source_buttons["processed"].isChecked()

    panel.record_btn.click()
    assert fake.recording and panel.record_btn.text() == "Stop"
    panel.record_btn.click()
    assert not fake.recording and panel.record_btn.text() == "Record"
    assert "Applying" in panel.preview_hint.text()
    assert not panel.play_btn.isEnabled()  # processed version not ready yet
    threads.pop()()
    panel._render_finished.emit()
    assert fake.renders == 1
    assert panel.play_btn.isEnabled()
    assert panel.seek_slider.maximum() == 3000
    assert panel.preview_time.text() == "0:00 / 0:03"

    panel.play_btn.click()
    assert fake.playing and panel.play_btn.text() == "Pause"
    fake.position = 48000 * 2
    panel._sync_preview()
    assert panel.seek_slider.value() == 2000
    assert panel.preview_time.text() == "0:02 / 0:03"
    panel.source_buttons["original"].click()
    assert fake.source == "original"
    panel.play_btn.click()
    assert not fake.playing

    # Tuning a voice effect re-renders the clip (debounced).
    panel.noise_slider.setValue(1)
    assert panel._render_timer.isActive()
    panel._render_timer.stop()
    panel._render_preview()
    panel._render_preview()  # a change during a render queues one more
    assert len(threads) == 1
    threads.pop()()
    panel._render_finished.emit()
    assert len(threads) == 1
    threads.pop()()
    panel._render_finished.emit()
    assert fake.renders == 3
    # Unrelated settings don't.
    panel.blur_slider.setValue(10)
    assert not panel._render_timer.isActive()


def test_voice_preview_auto_stops_at_limit(panel, monkeypatch):
    monkeypatch.setattr(
        tray.threading, "Thread",
        lambda target, **kwargs: SimpleNamespace(start=lambda: None),
    )
    fake = FakePreview()
    panel.voice_preview = fake
    panel.record_btn.click()
    fake.hit_limit = True
    panel._sync_preview()
    assert not fake.recording
    assert fake.has_clip
