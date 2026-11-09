from types import SimpleNamespace

import numpy as np
import pytest

from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline import preview as preview_module
from open_broadcast.pipeline.preview import VoicePreview


class CallbackStop(Exception):  # noqa: N818 (mirrors sounddevice)
    pass


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


@pytest.fixture
def backend(monkeypatch):
    streams = []

    def make(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    fake = SimpleNamespace(
        InputStream=make, OutputStream=make, CallbackStop=CallbackStop,
        default=SimpleNamespace(device=(0, 1)),
        query_devices=lambda index=None, kind=None: {"max_output_channels": 2, "index": 1},
        streams=streams,
    )
    monkeypatch.setattr(preview_module.audio, "sd", fake)
    monkeypatch.setattr(preview_module.audio, "wasapi_settings", lambda index: None)
    monkeypatch.setattr(
        preview_module.devices, "resolve",
        lambda kind, name: SimpleNamespace(index=0, name="Mic"),
    )
    return fake


@pytest.fixture
def preview(backend, monkeypatch):
    instance = VoicePreview(PipelineConfig(low_cut=False))
    monkeypatch.setattr(instance._pipeline, "_load_models", lambda: None)
    yield instance
    instance.close()


def feed(stream, samples):
    try:
        stream.callback(samples.reshape(-1, 1), len(samples), None, None)
    except CallbackStop:
        stream.active = False
        return True
    return False


def record(preview, backend, clip):
    preview.start_recording()
    stream = backend.streams[-1]
    for block in clip.reshape(-1, 512):
        feed(stream, block)
    preview.stop_recording()


def test_record_stop_and_render(preview, backend):
    clip = np.linspace(-0.2, 0.2, 512 * 20, dtype=np.float32)
    record(preview, backend, clip)
    assert backend.streams[0].closed
    assert preview.has_clip
    np.testing.assert_array_equal(preview.original, clip)
    assert preview.duration == pytest.approx(len(clip) / 48000)
    assert preview.processed is None
    preview.render()
    np.testing.assert_array_equal(preview.processed, clip)  # no models -> passthrough


def test_recording_stops_itself_at_limit(preview, backend, monkeypatch):
    monkeypatch.setattr(VoicePreview, "MAX_SECONDS", 1024 / 48000)
    preview.start_recording()
    stream = backend.streams[-1]
    assert not feed(stream, np.ones(512, np.float32))
    assert feed(stream, np.ones(512, np.float32))
    assert preview.recording_finished
    preview.stop_recording()
    assert len(preview.original) == 1024


def test_playback_switches_sources_without_jumping(preview, backend):
    clip = np.ones(512 * 4, dtype=np.float32)
    record(preview, backend, clip)
    preview.processed = clip * 0.25
    preview.set_source("original")
    preview.play()
    out_stream = backend.streams[-1]
    assert out_stream.kwargs["channels"] == 2
    out = np.empty((512, 2), np.float32)
    out_stream.callback(out, 512, None, None)
    assert np.all(out == 1.0)
    preview.set_source("processed")
    out_stream.callback(out, 512, None, None)
    assert np.all(out == 0.25)
    assert preview.position == 1024
    preview.seek(10)  # clamped to the end
    assert preview.position == len(clip)
    with pytest.raises(CallbackStop):
        out_stream.callback(out, 512, None, None)
    assert not preview.playing
    assert np.all(out == 0)
    preview.play()  # restarts from the beginning once finished
    assert preview.position == 0


def test_pause_closes_stream_and_keeps_position(preview, backend):
    record(preview, backend, np.ones(512 * 4, dtype=np.float32))
    preview.play()
    stream = backend.streams[-1]
    stream.callback(np.empty((512, 2), np.float32), 512, None, None)
    preview.pause()
    assert stream.closed and not preview.playing
    assert preview.position == 512


def test_errors_without_backend_or_mic(monkeypatch):
    monkeypatch.setattr(preview_module.audio, "sd", None)
    instance = VoicePreview(PipelineConfig())
    instance.start_recording()
    assert instance.error == "sounddevice/PortAudio not available"
    assert not instance.recording


def test_unknown_source_rejected(preview):
    with pytest.raises(ValueError):
        preview.set_source("both")


@pytest.mark.parametrize("operation", ["record", "play"])
def test_start_failure_closes_preview_stream(preview, backend, monkeypatch, operation):
    def fail_start(self):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(FakeStream, "start", fail_start)
    if operation == "record":
        preview.start_recording()
    else:
        preview.original = np.zeros(512, np.float32)
        preview.play()
    assert backend.streams[-1].closed
    assert not preview.recording
    assert not preview.playing
    assert "device unavailable" in preview.error
