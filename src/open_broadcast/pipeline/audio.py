"""Audio pipeline: mic -> DeepFilterNet3 (noise removal) -> Silero VAD gate -> virtual mic.

Both models run through ONNX Runtime, so no PyTorch is needed at runtime.
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import TYPE_CHECKING

import numpy as np

from open_broadcast import models
from open_broadcast.pipeline import devices

if TYPE_CHECKING:
    from open_broadcast.config import PipelineConfig

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None


def _ort_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class DeepFilterNet:
    """Streaming DeepFilterNet3 (48 kHz full-band). Processes fixed 512-sample hops.

    Adds fft_size - hop_size = 512 samples (~10.7 ms) of algorithmic latency.
    """

    SAMPLE_RATE = 48000
    HOP_SIZE = 512
    NB_ERB = 32
    NB_DF = 96

    def __init__(self, model_path=None):
        self._session = _ort_session(model_path or models.model_path("deepfilternet"))
        inputs = self._session.get_inputs()
        self._state_names = [i.name for i in inputs[1:]]
        self._state_shapes = {i.name: i.shape for i in inputs[1:]}
        self.reset()

    def reset(self) -> None:
        self._states = {n: np.zeros(s, dtype=np.float32) for n, s in self._state_shapes.items()}
        self._states["erb_norm_state"] = np.linspace(-60.0, -90.0, self.NB_ERB, dtype=np.float32)
        self._states["band_unit_norm_state"] = np.linspace(
            0.001, 0.0001, self.NB_DF, dtype=np.float32
        ).reshape(1, self.NB_DF, 1)

    def process_hop(self, frame: np.ndarray) -> np.ndarray:
        out = self._session.run(None, {"input_frame": frame, **self._states})
        self._states = dict(zip(self._state_names, out[1:], strict=True))
        return out[0]

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process any multiple of HOP_SIZE samples."""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) % self.HOP_SIZE:
            raise ValueError(f"Length must be a multiple of {self.HOP_SIZE}")
        if len(audio) == self.HOP_SIZE:
            return self.process_hop(audio)
        return np.concatenate(
            [self.process_hop(audio[i:i + self.HOP_SIZE])
             for i in range(0, len(audio), self.HOP_SIZE)]
        )


class SileroVAD:
    """Silero VAD v5/v6 on ONNX Runtime. Consumes 512-sample windows at 16 kHz."""

    SAMPLE_RATE = 16000
    WINDOW = 512
    CONTEXT = 64

    def __init__(self, model_path=None):
        self._session = _ort_session(model_path or models.model_path("silero_vad"))
        self._sr = np.array(self.SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)

    def probability(self, window: np.ndarray) -> float:
        x = np.concatenate([self._context, window.astype(np.float32)])[None, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[0, -self.CONTEXT:]
        return float(out[0, 0])


class SpeechGate:
    """Turns per-window VAD probabilities into a smooth gain envelope."""

    def __init__(self, sample_rate: int, holdover_s: float = 0.3, floor: float = 0.0,
                 attack_s: float = 0.005, release_s: float = 0.08):
        self.sample_rate = sample_rate
        self.holdover_samples = int(holdover_s * sample_rate)
        self.floor = floor
        self.attack_step = 1.0 / max(1, int(attack_s * sample_rate))
        self.release_step = 1.0 / max(1, int(release_s * sample_rate))
        self.gain = 1.0
        self._since_speech = 0
        self.speech_active = True

    def update(self, is_speech: bool, samples: int) -> None:
        if is_speech:
            self._since_speech = 0
            self.speech_active = True
        else:
            self._since_speech += samples
            if self._since_speech > self.holdover_samples:
                self.speech_active = False

    def apply(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        target = 1.0 if self.speech_active else self.floor
        if target > self.gain:
            ramp = np.minimum(target, self.gain + self.attack_step * np.arange(1, n + 1))
        elif target < self.gain:
            ramp = np.maximum(target, self.gain - self.release_step * np.arange(1, n + 1))
        else:
            return audio * target if target != 1.0 else audio
        self.gain = float(ramp[-1])
        return audio * ramp.astype(np.float32)


class AudioPipeline:
    """Captures mic input, denoises it, gates silence, and writes to the virtual mic."""

    SAMPLE_RATE = 48000
    BLOCK_SIZE = DeepFilterNet.HOP_SIZE  # 10.7 ms

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stream = None
        self._lock = threading.Lock()
        self._dfn: DeepFilterNet | None = None
        self._vad: SileroVAD | None = None
        self._gate = SpeechGate(self.SAMPLE_RATE)
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._resample_buf = np.zeros(0, dtype=np.float32)
        self._errors_logged = 0
        self.level = 0.0
        self.error: str | None = None

    @property
    def speech_active(self) -> bool:
        return self._gate.speech_active

    def _load_models(self) -> None:
        try:
            self._dfn = DeepFilterNet()
            log.info("DeepFilterNet3 loaded")
        except Exception as e:
            log.error("DeepFilterNet unavailable: %s", e)
        try:
            self._vad = SileroVAD()
            log.info("Silero VAD loaded")
        except Exception as e:
            log.error("Silero VAD unavailable: %s", e)

    def process_block(self, audio: np.ndarray) -> np.ndarray:
        """Process one mono float32 block (BLOCK_SIZE samples). Pure function of state."""
        cfg = self.config
        with self._lock:
            if cfg.noise_suppression and self._dfn is not None:
                processed = self._dfn.process(audio)
            else:
                processed = audio

            if cfg.vad_enabled and self._vad is not None:
                # 48k -> 16k: average groups of 3 (cheap low-pass + decimate).
                resample_input = np.concatenate([self._resample_buf, processed])
                usable = len(resample_input) - len(resample_input) % 3
                down = resample_input[:usable].reshape(-1, 3).mean(axis=1)
                self._resample_buf = resample_input[usable:].copy()
                self._vad_buf = np.concatenate([self._vad_buf, down])
                while len(self._vad_buf) >= SileroVAD.WINDOW:
                    window = self._vad_buf[:SileroVAD.WINDOW]
                    self._vad_buf = self._vad_buf[SileroVAD.WINDOW:]
                    prob = self._vad.probability(window)
                    self._gate.update(prob >= cfg.vad_threshold, SileroVAD.WINDOW * 3)
            else:
                self._vad_buf = np.zeros(0, dtype=np.float32)
                self._resample_buf = np.zeros(0, dtype=np.float32)
                self._gate.update(True, len(processed))

            return self._gate.apply(processed)

    def start(self) -> None:
        self.error = None
        if sd is None:
            self.error = "sounddevice/PortAudio not available"
            log.error(self.error)
            return

        self._load_models()
        in_dev = devices.resolve("input", self.config.input_device)
        out_dev = devices.resolve("output", self.config.output_device)
        if in_dev is None or out_dev is None:
            self.error = "No audio input/output device found"
            log.error(self.error)
            return

        out_channels = min(2, out_dev.max_output_channels)

        def callback(indata, outdata, frames, time_info, status):
            if status:
                log.debug("audio status: %s", status)
            try:
                mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
                self.level = float(np.sqrt(np.mean(mono * mono)))
                if frames != self.BLOCK_SIZE:
                    # Shouldn't happen with a fixed blocksize; pass through untouched.
                    out = mono
                else:
                    out = self.process_block(np.ascontiguousarray(mono, dtype=np.float32))
                outdata[:] = out[:, None]
            except Exception:
                outdata.fill(0)
                if self._errors_logged < 5:
                    self._errors_logged += 1
                    log.exception("Audio callback error")

        extra = None
        if platform.system() == "Windows" and "WASAPI" in sd.query_hostapis(
            sd.query_devices(in_dev.index)["hostapi"]
        )["name"]:
            # Let WASAPI resample when the device isn't running at 48 kHz.
            extra = (sd.WasapiSettings(auto_convert=True), sd.WasapiSettings(auto_convert=True))

        try:
            self._stream = sd.Stream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                channels=(1, out_channels),
                dtype="float32",
                latency="low",
                device=(in_dev.index, out_dev.index),
                callback=callback,
                extra_settings=extra,
            )
            self._stream.start()
            log.info("Audio running: %r -> %r", in_dev.name, out_dev.name)
        except Exception as e:
            self._stream = None
            self.error = f"Failed to open audio devices: {e}"
            log.error(self.error)
            self._log_platform_hint()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.exception("Error closing audio stream")
            self._stream = None
        with self._lock:
            if self._dfn:
                self._dfn.reset()
            if self._vad:
                self._vad.reset()
            self._vad_buf = np.zeros(0, dtype=np.float32)
            self._resample_buf = np.zeros(0, dtype=np.float32)
            self._gate = SpeechGate(self.SAMPLE_RATE)
            self.level = 0.0

    def update_config(self, config: PipelineConfig) -> None:
        self.config = config

    @staticmethod
    def _log_platform_hint() -> None:
        system = platform.system()
        if system == "Linux":
            log.info("Hint: create a virtual mic with "
                     "`pactl load-module module-null-sink sink_name=OpenBroadcast`")
        elif system == "Windows":
            log.info("Hint: install VB-Audio Virtual Cable (https://vb-audio.com/Cable/)")
        elif system == "Darwin":
            log.info("Hint: install BlackHole (`brew install blackhole-2ch`)")
