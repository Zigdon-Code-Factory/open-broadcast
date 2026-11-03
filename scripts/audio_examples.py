"""Download attributed real speech/noise examples and create noisy/processed listening pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_diagnostics import HOP, SAMPLE_RATE, rms, write_wav  # noqa: E402

from open_broadcast.config import PipelineConfig  # noqa: E402
from open_broadcast.pipeline.audio import AudioPipeline  # noqa: E402

SPEECH_API = (
    "https://datasets-server.huggingface.co/rows?dataset=openslr%2Flibrispeech_asr"
    "&config=clean&split=test&length=1&offset="
)
NOISE_FILES = {
    "rain": "1-17367-A-10.wav",
    "clock_tick": "1-21934-A-38.wav",
    "helicopter": "1-172649-A-40.wav",
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "OpenBroadcast-audio-tests"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def save_download(url, path):
    content = fetch(url)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def read_audio(path):
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    positions = np.arange(int(len(mono) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE
    return np.interp(positions, np.arange(len(mono)), mono).astype(np.float32)


def process_example(speech, noise, snr_db, output_dir, name):
    speech = speech[:SAMPLE_RATE * 15]
    clean = np.concatenate([np.zeros(SAMPLE_RATE), speech, np.zeros(SAMPLE_RATE)])
    clean = np.pad(clean, (0, -len(clean) % HOP)).astype(np.float32)
    background = np.resize(noise, len(clean)).astype(np.float32)
    background *= rms(speech) / max(rms(noise), 1e-12) / (10 ** (snr_db / 20))
    mixture = clean + background
    scale = min(1.0, 0.95 / max(float(np.max(np.abs(mixture))), 1e-12))
    mixture *= scale
    pipeline = AudioPipeline(PipelineConfig())
    pipeline._load_models()
    if pipeline._dfn is None or pipeline._vad is None:
        raise RuntimeError("Both audio models must be available")
    blocks = []
    active = 0
    for block in mixture.reshape(-1, HOP):
        blocks.append(pipeline.process_block(block))
        active += int(pipeline.speech_active)
    processed = np.concatenate(blocks)
    write_wav(output_dir / f"{name}-noisy.wav", mixture)
    write_wav(output_dir / f"{name}-processed.wav", processed)
    speech_output = processed[SAMPLE_RATE:SAMPLE_RATE + len(speech)]
    return {"name": name, "snr_db": snr_db, "normalization_gain": scale,
            "duration_seconds": len(mixture) / SAMPLE_RATE,
            "speech_output_rms": rms(speech_output),
            "speech_active_fraction": active / len(blocks),
            "noisy_file": f"{name}-noisy.wav", "processed_file": f"{name}-processed.wav",
            "passed": bool(np.isfinite(processed).all() and rms(speech_output) > 1e-4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/audio-tests/online")
    args = parser.parse_args()
    output_dir = args.output_dir
    originals = output_dir / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    revision = json.loads(fetch(
        "https://api.github.com/repos/karolpiczak/ESC-50/commits/master"
    ))["sha"]
    noise_root = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{revision}"
    save_download(f"{noise_root}/LICENSE", output_dir / "ESC-50-LICENSE.txt")
    manifest = {
        "sources": [], "cases": [],
        "limitations": (
            "Automated integrity/activity checks, not perceptual quality scores. "
            "Noise is tiled; clips include one second of noise before/after speech."
        ),
    }
    noises = {}
    for category, filename in NOISE_FILES.items():
        path = originals / filename
        url = f"{noise_root}/audio/{filename}"
        digest = save_download(url, path)
        noises[category] = read_audio(path)
        manifest["sources"].append({"kind": "noise", "category": category,
                                    "file": f"originals/{filename}", "source_url": url,
                                    "collection": "ESC-10 subset of ESC-50",
                                    "license": "CC BY 3.0; see ESC-50-LICENSE.txt for attribution",
                                    "sha256": digest})
    for offset in (0, 1000, 2000):
        row_url = SPEECH_API + str(offset)
        row = json.loads(fetch(row_url))["rows"][0]["row"]
        identifier = row["id"]
        if not re.fullmatch(r"[0-9-]+", identifier):
            raise ValueError("Unexpected speech sample identifier")
        path = originals / f"{identifier}.flac"
        digest = save_download(row["audio"][0]["src"], path)
        speech = read_audio(path)
        write_wav(originals / f"{identifier}.wav", speech)
        manifest["sources"].append({"kind": "speech", "id": identifier,
                                    "speaker_id": row["speaker_id"], "text": row["text"],
                                    "file": f"originals/{identifier}.flac",
                                    "source_url": row_url, "collection": "LibriSpeech test-clean",
                                    "license": "CC BY 4.0",
                                    "dataset_url": "https://www.openslr.org/12", "sha256": digest})
        for category, noise in noises.items():
            for snr_db in (0, 10):
                name = f"{identifier}-{category}-{snr_db}dB"
                result = process_example(speech, noise, snr_db, output_dir, name)
                manifest["cases"].append(result)
                print(f"{'PASS' if result['passed'] else 'FAIL'} {name}", flush=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    manifest["passed"] = all(case["passed"] for case in manifest["cases"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"Saved {len(manifest['sources'])} original clips and {len(manifest['cases'])} pairs")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
