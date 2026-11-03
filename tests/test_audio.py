import numpy as np

from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline.audio import AudioPipeline, DeepFilterNet, SileroVAD, SpeechGate

SR = 48000


def _noisy_tone(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 1 * t) > 0)
    noisy = clean + 0.05 * rng.standard_normal(len(t))
    n = len(noisy) - len(noisy) % 512
    return clean[:n].astype(np.float32), noisy[:n].astype(np.float32)


def test_gate_closes_after_holdover_and_reopens():
    g = SpeechGate(SR, holdover_s=0.1, floor=0.0, release_s=0.01)
    block = np.ones(512, np.float32)
    assert np.allclose(g.apply(block), 1.0)
    for _ in range(20):
        g.update(False, 512)
        out = g.apply(block)
    assert not g.speech_active
    assert np.allclose(out, 0.0)
    g.update(True, 512)
    out = g.apply(block)
    assert g.speech_active and out[-1] > out[0]


def test_deepfilternet_reduces_noise(require_model):
    require_model("deepfilternet")
    clean, noisy = _noisy_tone()
    dfn = DeepFilterNet()
    out = dfn.process(noisy)
    assert out.shape == noisy.shape and out.dtype == np.float32
    silent = clean == 0
    # Skip the first 0.25 s while the model's normalisation state warms up.
    silent[: SR // 4] = False
    assert np.sqrt(np.mean(out[silent] ** 2)) < 0.25 * np.sqrt(np.mean(noisy[silent] ** 2))


def test_vad_low_for_noise(require_model):
    require_model("silero_vad")
    vad = SileroVAD()
    rng = np.random.default_rng(1)
    probs = [vad.probability((0.01 * rng.standard_normal(512)).astype(np.float32))
             for _ in range(20)]
    assert max(probs) < 0.5


def test_pipeline_process_block(require_model):
    require_model("deepfilternet")
    require_model("silero_vad")
    p = AudioPipeline(PipelineConfig())
    p._load_models()
    _, noisy = _noisy_tone(0.5)
    out = np.concatenate([p.process_block(noisy[i:i + 512]) for i in range(0, len(noisy), 512)])
    assert out.shape == noisy.shape
    assert np.isfinite(out).all()
