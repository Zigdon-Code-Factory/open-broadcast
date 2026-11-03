from types import SimpleNamespace

import numpy as np
import pytest

from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline import audio as audio_module
from open_broadcast.pipeline.audio import AudioPipeline, DeepFilterNet, SileroVAD, SpeechGate


class RecordingVAD:
    def __init__(self):
        self.windows = []

    def probability(self, window):
        self.windows.append(window.copy())
        return 0.0

    def reset(self):
        self.windows.clear()


def test_vad_resampling_preserves_samples_across_blocks():
    pipeline = AudioPipeline(PipelineConfig(noise_suppression=False))
    pipeline._vad = RecordingVAD()
    source = np.arange(512 * 6, dtype=np.float32)
    for offset in range(0, len(source), 512):
        pipeline.process_block(source[offset:offset + 512])
    assert len(pipeline._vad.windows) == 2
    actual = np.concatenate(pipeline._vad.windows)
    np.testing.assert_array_equal(actual, source.reshape(-1, 3).mean(axis=1))


def test_stop_resets_gate_for_next_session():
    pipeline = AudioPipeline(PipelineConfig())
    pipeline._gate.update(False, 48000)
    pipeline._gate.apply(np.ones(512 * 10, dtype=np.float32))
    assert not pipeline.speech_active
    pipeline.stop()
    assert pipeline.speech_active
    assert pipeline._gate.gain == 1.0
    assert len(pipeline._vad_buf) == 0
    assert len(pipeline._resample_buf) == 0


def test_disabling_vad_discards_buffered_audio():
    pipeline = AudioPipeline(PipelineConfig(noise_suppression=False))
    pipeline._vad = RecordingVAD()
    pipeline.process_block(np.ones(512, dtype=np.float32))
    assert len(pipeline._resample_buf) == 2
    assert len(pipeline._vad_buf) == 170
    pipeline.update_config(PipelineConfig(noise_suppression=False, vad_enabled=False))
    pipeline.process_block(np.zeros(512, dtype=np.float32))
    assert len(pipeline._resample_buf) == 0
    assert len(pipeline._vad_buf) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_models_fall_back_without_corrupting_audio(enabled):
    pipeline = AudioPipeline(PipelineConfig(noise_suppression=enabled, vad_enabled=enabled))
    source = np.linspace(-0.8, 0.8, 512, dtype=np.float32)
    np.testing.assert_array_equal(pipeline.process_block(source), source)


def test_disabled_processing_bypasses_loaded_models():
    def unexpected_call(*args):
        pytest.fail("Disabled processing must not run models")

    pipeline = AudioPipeline(PipelineConfig(noise_suppression=False, vad_enabled=False))
    pipeline._dfn = SimpleNamespace(process=unexpected_call)
    pipeline._vad = SimpleNamespace(probability=unexpected_call)
    source = np.ones(512, dtype=np.float32) * 0.2
    np.testing.assert_array_equal(pipeline.process_block(source), source)


def test_gate_holdover_preserves_end_of_speech():
    gate = SpeechGate(48000, holdover_s=0.3)
    gate.update(False, 14400)
    assert gate.speech_active
    gate.update(False, 1)
    assert not gate.speech_active


def test_gate_release_and_attack_are_continuous_and_bounded():
    gate = SpeechGate(48000, holdover_s=0, floor=0.05)
    block = np.ones(512, dtype=np.float32)
    gate.update(False, 512)
    release = np.concatenate([gate.apply(block) for _ in range(10)])
    assert np.all(np.diff(release) <= 1e-7)
    assert release.min() == pytest.approx(0.05)
    assert np.max(np.abs(np.diff(release))) <= gate.release_step + 1e-7
    gate.update(True, 512)
    attack = gate.apply(block)
    assert np.all(np.diff(attack) >= -1e-7)
    assert attack.max() == pytest.approx(1.0)
    assert np.max(np.abs(np.diff(attack))) <= gate.attack_step + 1e-7


def test_deepfilternet_streamed_and_batched_output_match(require_model):
    require_model("deepfilternet")
    model = DeepFilterNet()
    source = np.random.default_rng(7).normal(0, 0.08, 512 * 30).astype(np.float32)
    batched = model.process(source)
    model.reset()
    streamed = np.concatenate([model.process(block) for block in source.reshape(-1, 512)])
    np.testing.assert_allclose(streamed, batched, atol=1e-7, rtol=0)


def test_deepfilternet_rejects_incomplete_hop(require_model):
    require_model("deepfilternet")
    with pytest.raises(ValueError, match="multiple of 512"):
        DeepFilterNet().process(np.zeros(511, dtype=np.float32))


@pytest.mark.parametrize("kind", ["silence", "white_noise", "hum", "impulses"])
def test_model_is_finite_and_stable_for_signal_types(require_model, kind):
    require_model("deepfilternet")
    samples = 512 * 100
    time = np.arange(samples) / 48000
    source = np.zeros(samples, dtype=np.float32)
    if kind == "white_noise":
        source = np.random.default_rng(8).uniform(-0.95, 0.95, samples).astype(np.float32)
    elif kind == "hum":
        source = (0.2 * np.sin(2 * np.pi * 60 * time)).astype(np.float32)
    elif kind == "impulses":
        source[::4800] = 0.9
    output = DeepFilterNet().process(source)
    assert output.shape == source.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) < 2.0
    if kind == "silence":
        assert np.max(np.abs(output)) < 1e-5


def test_vad_reset_reproduces_probabilities(require_model):
    require_model("silero_vad")
    model = SileroVAD()
    windows = np.random.default_rng(12).normal(0, 0.02, (20, 512)).astype(np.float32)
    before = [model.probability(window) for window in windows]
    model.reset()
    after = [model.probability(window) for window in windows]
    np.testing.assert_allclose(after, before, atol=1e-7, rtol=0)
    assert all(0 <= probability <= 1 for probability in after)


def test_pipeline_gates_sustained_silence(require_model):
    require_model("deepfilternet")
    require_model("silero_vad")
    pipeline = AudioPipeline(PipelineConfig())
    pipeline._load_models()
    assert pipeline._dfn is not None and pipeline._vad is not None
    for _ in range(150):
        output = pipeline.process_block(np.zeros(512, dtype=np.float32))
    assert not pipeline.speech_active
    assert np.max(np.abs(output)) < 1e-6


def test_missing_audio_backend_reports_error(monkeypatch):
    monkeypatch.setattr(audio_module, "sd", None)
    pipeline = AudioPipeline(PipelineConfig())
    pipeline.start()
    assert pipeline.error == "sounddevice/PortAudio not available"


@pytest.fixture
def callback_pipeline(monkeypatch):
    class FakeStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            self.closed = True

    backend = SimpleNamespace(
        Stream=FakeStream,
        query_devices=lambda index: {"hostapi": 0},
        query_hostapis=lambda index: {"name": "Test backend"},
    )
    monkeypatch.setattr(audio_module, "sd", backend)
    monkeypatch.setattr(
        audio_module.devices, "resolve",
        lambda kind, name: SimpleNamespace(index=0, name=kind, max_output_channels=2),
    )
    pipeline = AudioPipeline(PipelineConfig(noise_suppression=False, vad_enabled=False))
    monkeypatch.setattr(pipeline, "_load_models", lambda: None)
    pipeline.start()
    assert pipeline.error is None
    yield pipeline
    pipeline.stop()


def test_audio_callback_duplicates_mono_to_stereo(callback_pipeline):
    source = np.linspace(-0.3, 0.3, 512, dtype=np.float32).reshape(-1, 1)
    output = np.empty((512, 2), dtype=np.float32)
    callback_pipeline._stream.callback(source, output, 512, None, None)
    np.testing.assert_array_equal(output[:, 0], source[:, 0])
    np.testing.assert_array_equal(output[:, 1], source[:, 0])
    assert callback_pipeline.level == pytest.approx(np.sqrt(np.mean(source ** 2)))


def test_audio_callback_outputs_silence_on_processing_failure(callback_pipeline, monkeypatch):
    def fail_processing(samples):
        raise RuntimeError("Test processing failure")

    monkeypatch.setattr(callback_pipeline, "process_block", fail_processing)
    output = np.ones((512, 2), dtype=np.float32)
    callback_pipeline._stream.callback(np.ones((512, 1), np.float32), output, 512, None, None)
    assert np.count_nonzero(output) == 0
    assert callback_pipeline._errors_logged == 1


def test_stopping_audio_closes_stream(callback_pipeline):
    stream = callback_pipeline._stream
    assert stream.started
    callback_pipeline.stop()
    assert not stream.started
    assert stream.closed
    assert callback_pipeline._stream is None
