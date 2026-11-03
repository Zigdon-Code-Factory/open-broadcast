"""Run repeatable audio diagnostics; optionally test the VB-CABLE route without a microphone."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_broadcast.config import PipelineConfig  # noqa: E402
from open_broadcast.pipeline import devices  # noqa: E402
from open_broadcast.pipeline.audio import AudioPipeline, DeepFilterNet, SileroVAD  # noqa: E402

SAMPLE_RATE = 48000
HOP = 512


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def write_wav(path, samples):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def read_speech(path):
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("Speech WAV must contain 16-bit PCM")
        rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels).mean(axis=1) / 32768
    positions = np.arange(int(len(samples) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def noise_tests():
    rng = np.random.default_rng(42)
    length = HOP * 400
    timeline = np.arange(length) / SAMPLE_RATE
    white = rng.normal(0, 0.05, length)
    fan = np.convolve(white, np.ones(32) / 32, mode="same")
    hum = 0.05 * np.sin(2 * np.pi * 60 * timeline)
    hum += 0.02 * np.sin(2 * np.pi * 120 * timeline)
    results = []
    model = DeepFilterNet()
    for name, samples in [("white", white), ("fan", fan), ("mains_hum", hum)]:
        model.reset()
        samples = samples.astype(np.float32)
        processed = model.process(samples)
        input_rms = rms(samples[SAMPLE_RATE:])
        output_rms = rms(processed[SAMPLE_RATE:])
        reduction = float(20 * np.log10(input_rms / max(output_rms, 1e-12)))
        results.append({"name": name, "reduction_db": reduction,
                        "passed": bool(reduction >= 6 and np.isfinite(processed).all())})
    return results


def benchmark():
    pipeline = AudioPipeline(PipelineConfig())
    pipeline._load_models()
    if pipeline._dfn is None or pipeline._vad is None:
        raise RuntimeError("Both audio models are required for benchmarking")
    rng = np.random.default_rng(3)
    blocks = rng.normal(0, 0.05, (2000, HOP)).astype(np.float32)
    durations = []
    for index, block in enumerate(blocks):
        started = time.perf_counter()
        pipeline.process_block(block)
        elapsed = (time.perf_counter() - started) * 1000
        if index >= 50:
            durations.append(elapsed)
    budget = HOP / SAMPLE_RATE * 1000
    return {"blocks": len(durations), "audio_seconds": len(blocks) * HOP / SAMPLE_RATE,
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)), "max_ms": max(durations),
            "block_budget_ms": budget, "overruns": sum(value > budget for value in durations),
            "passed": bool(np.percentile(durations, 95) < budget)}


def speech_test(path, output_dir):
    speech = read_speech(path)
    clean = np.concatenate([np.zeros(SAMPLE_RATE // 2), speech, np.zeros(SAMPLE_RATE)])
    clean = np.pad(clean, (0, -len(clean) % HOP)).astype(np.float32)
    noisy = clean + np.random.default_rng(4).normal(0, 0.025, len(clean)).astype(np.float32)
    pipeline = AudioPipeline(PipelineConfig())
    pipeline._load_models()
    if pipeline._dfn is None or pipeline._vad is None:
        raise RuntimeError("Both audio models are required for the speech test")
    output = np.concatenate([pipeline.process_block(block) for block in noisy.reshape(-1, HOP)])
    vad = SileroVAD()
    downsampled = speech[:len(speech) - len(speech) % 3].reshape(-1, 3).mean(axis=1)
    probabilities = [vad.probability(downsampled[offset:offset + 512])
                     for offset in range(0, len(downsampled) - 511, 512)]
    detected = sum(probability >= 0.5 for probability in probabilities)
    tail_rms = rms(output[-SAMPLE_RATE // 4:])
    speech_rms = rms(output[SAMPLE_RATE // 2:SAMPLE_RATE // 2 + len(speech)])
    write_wav(output_dir / "speech-noisy.wav", noisy)
    write_wav(output_dir / "speech-processed.wav", output)
    return {"duration_seconds": len(speech) / SAMPLE_RATE,
            "speech_windows": detected, "vad_windows": len(probabilities),
            "peak_vad_probability": max(probabilities, default=0),
            "processed_speech_rms": speech_rms, "silent_tail_rms": tail_rms,
            "passed": bool(detected >= 3 and speech_rms > 1e-4 and tail_rms < 1e-5
                           and np.isfinite(output).all())}


def cable_test():
    import sounddevice as sd

    input_device = next((device for device in devices.list_devices("input")
                         if device.name.startswith("CABLE Output (")), None)
    output_device = next((device for device in devices.list_devices("output")
                          if device.name.startswith("CABLE Input (")), None)
    if input_device is None or output_device is None:
        return {"passed": False, "error": "VB-CABLE input/output pair not found"}
    timeline = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    signal = (0.03 * np.sin(2 * np.pi * (300 * timeline + 400 * timeline ** 2)))
    signal[:480] *= np.linspace(0, 1, 480)
    signal[-480:] *= np.linspace(1, 0, 480)
    signal = signal.astype(np.float32)
    playback = np.pad(signal, (SAMPLE_RATE // 2, SAMPLE_RATE))
    captured = []
    statuses = []
    finished = threading.Event()
    position = 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal position
        captured.append(indata.copy())
        if status:
            statuses.append(str(status))
        outdata.fill(0)
        count = min(frames, len(playback) - position)
        if count > 0:
            outdata[:count] = playback[position:position + count, None]
            position += count
        if position >= len(playback):
            raise sd.CallbackStop

    extra = (sd.WasapiSettings(auto_convert=True), sd.WasapiSettings(auto_convert=True))
    with sd.Stream(device=(input_device.index, output_device.index), samplerate=SAMPLE_RATE,
                   channels=(2, 2), blocksize=HOP, dtype="float32", latency="low",
                   extra_settings=extra, callback=callback, finished_callback=finished.set):
        if not finished.wait(12):
            raise RuntimeError("VB-CABLE test timed out")
    recording = np.concatenate(captured)
    mono = recording.mean(axis=1)
    fft_size = 1 << (len(mono) + len(signal) - 2).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(mono, fft_size) * np.fft.rfft(signal[::-1], fft_size), fft_size
    )[:len(mono) + len(signal) - 1]
    lag = int(np.argmax(correlation)) - len(signal) + 1
    aligned = mono[max(lag, 0):max(lag, 0) + len(signal)]
    reference = signal[max(-lag, 0):max(-lag, 0) + len(aligned)]
    score = float(np.dot(aligned, reference)
                  / max(np.linalg.norm(aligned) * np.linalg.norm(reference), 1e-12))
    return {"input": input_device.name, "output": output_device.name,
            "correlation": score, "round_trip_ms": (lag - SAMPLE_RATE // 2) / SAMPLE_RATE * 1000,
            "status_events": statuses, "captured_frames": len(mono),
            "passed": bool(score > 0.95 and np.isfinite(recording).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-file", type=Path)
    parser.add_argument("--device-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "audio-tests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"noise": noise_tests(), "benchmark": benchmark()}
    if args.speech_file:
        report["speech"] = speech_test(args.speech_file, args.output_dir)
    if args.device_test:
        try:
            report["virtual_cable"] = cable_test()
        except Exception as error:
            report["virtual_cable"] = {"passed": False, "error": str(error)}
    checks = report["noise"] + [value for value in report.values() if isinstance(value, dict)]
    report["passed"] = all(check["passed"] for check in checks)
    text = json.dumps(report, indent=2)
    (args.output_dir / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
