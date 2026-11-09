import json

import numpy as np
import pytest

from open_broadcast import config as config_mod
from open_broadcast.config import NOISE_REDUCTION_MAX_DB, PipelineConfig
from open_broadcast.pipeline.audio import AudioPipeline, DeepFilterNet, HighPass, attenuation_mix

SR = 48000


def tone(freq, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def test_attenuation_mix_matches_db_limit():
    assert attenuation_mix(NOISE_REDUCTION_MAX_DB) == 0.0
    assert attenuation_mix(20) == pytest.approx(0.1)
    assert attenuation_mix(6) == pytest.approx(0.501, abs=1e-3)
    assert attenuation_mix(-5) == 1.0


def test_high_pass_cuts_rumble_and_keeps_voice():
    hpf = HighPass(SR, 80)
    rumble = hpf.process(tone(30))[SR // 2:]
    hpf.reset()
    voice = hpf.process(tone(300))[SR // 2:]
    assert rms(rumble) < 0.2 * rms(tone(30))
    assert rms(voice) == pytest.approx(rms(tone(300)), rel=0.03)


def test_high_pass_streaming_matches_single_pass():
    signal = np.random.default_rng(0).standard_normal(512 * 8).astype(np.float32)
    whole = HighPass(SR, 120).process(signal)
    streamed = HighPass(SR, 120)
    blocks = np.concatenate([streamed.process(b) for b in signal.reshape(-1, 512)])
    np.testing.assert_allclose(blocks, whole, atol=1e-6)


def test_low_cut_toggle_in_pipeline():
    cfg = PipelineConfig(noise_suppression=False, vad_enabled=False, low_cut=True)
    pipeline = AudioPipeline(cfg)
    out = pipeline.render(tone(25))
    assert rms(out[SR // 2:]) < 0.1 * rms(tone(25))
    cfg.low_cut = False
    np.testing.assert_array_equal(pipeline.render(tone(25)), tone(25))


def test_render_without_models_is_identity_and_aligned():
    pipeline = AudioPipeline(PipelineConfig(low_cut=False))
    clip = np.random.default_rng(1).standard_normal(12345).astype(np.float32) * 0.1
    out = pipeline.render(clip)
    assert len(out) == len(clip)
    np.testing.assert_array_equal(out, clip)


class DelayModel:
    """Stands in for DeepFilterNet: delays by LATENCY and scales by 0.5."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = np.zeros(DeepFilterNet.LATENCY, dtype=np.float32)

    def process(self, audio):
        joined = np.concatenate([self._buf, audio])
        out, self._buf = joined[:len(audio)], joined[len(audio):]
        return out * 0.5


def test_strength_blend_lines_up_dry_signal_with_model_output():
    cfg = PipelineConfig(vad_enabled=False, noise_reduction_db=20)
    pipeline = AudioPipeline(cfg)
    pipeline._dfn = DelayModel()
    clip = np.random.default_rng(2).standard_normal(SR // 2).astype(np.float32) * 0.1
    out = pipeline.render(clip)
    # 0.9 * (0.5 * x) + 0.1 * x, with no comb filtering from a misaligned dry path.
    np.testing.assert_allclose(out, clip * 0.55, atol=1e-6)
    cfg.noise_reduction_db = NOISE_REDUCTION_MAX_DB
    np.testing.assert_allclose(pipeline.render(clip), clip * 0.5, atol=1e-6)


def test_holdover_setting_applies_to_live_gate():
    cfg = PipelineConfig(noise_suppression=False, vad_holdover=0.3)
    pipeline = AudioPipeline(cfg)
    cfg.vad_holdover = 1.2
    pipeline.process_block(np.zeros(512, dtype=np.float32))
    assert pipeline._gate.holdover_samples == int(1.2 * SR)


def test_deepfilternet_latency_constant(require_model):
    require_model("deepfilternet")
    dfn = DeepFilterNet()
    rng = np.random.default_rng(3)
    t = np.arange(SR) / SR
    # Speech-like harmonic signal: DFN passes it largely intact, so correlation is clean.
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 720), 1))
    voice = (0.2 * voice * (1 + np.sin(2 * np.pi * 3 * t)) +
             0.005 * rng.standard_normal(SR)).astype(np.float32)
    out = np.concatenate([dfn.process(b) for b in voice[:SR // 512 * 512].reshape(-1, 512)])
    src = voice[:len(out)]
    lags = range(0, 4 * 512)
    best = max(lags, key=lambda lag: float(np.dot(src[:len(src) - lag], out[lag:])))
    assert abs(best - DeepFilterNet.LATENCY) <= 8


def test_config_clamps_tuning_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"noise_reduction_db": 500, "vad_holdover": 0,
                                "low_cut_hz": 5000}))
    cfg = config_mod.load(path)
    assert cfg.noise_reduction_db == NOISE_REDUCTION_MAX_DB
    assert cfg.vad_holdover == 0.05
    assert cfg.low_cut_hz == 300.0


def test_config_resets_invalid_tuning_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"noise_reduction_db": "loud", "vad_holdover": 0.5}))
    cfg = config_mod.load(path)
    defaults = PipelineConfig()
    assert cfg.noise_reduction_db == defaults.noise_reduction_db
    assert cfg.vad_holdover == defaults.vad_holdover
