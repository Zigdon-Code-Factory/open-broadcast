"""Voice preview: record a short clip, then play it back raw or with the current effects.

Uses its own AudioPipeline instance, so rendering never disturbs the live stream, and
plays through the system default output (your speakers), not the virtual cable.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from open_broadcast.pipeline import audio, devices
from open_broadcast.pipeline.audio import AudioPipeline

if TYPE_CHECKING:
    from open_broadcast.config import PipelineConfig

log = logging.getLogger(__name__)

SOURCES = ("original", "processed")


class VoicePreview:
    SAMPLE_RATE = AudioPipeline.SAMPLE_RATE
    MAX_SECONDS = 10.0

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipeline = AudioPipeline(config)
        self._models_loaded = False
        self._render_lock = threading.Lock()
        self._record_stream = None
        self._play_stream = None
        self._chunks: list[np.ndarray] = []
        self._recorded = 0
        self.original: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self.position = 0
        self.source = "processed"
        self.recording = False
        self.playing = False
        self.error: str | None = None

    # --- state --------------------------------------------------------------------------------

    @property
    def has_clip(self) -> bool:
        return self.original is not None and len(self.original) > 0

    @property
    def duration(self) -> float:
        return len(self.original) / self.SAMPLE_RATE if self.has_clip else 0.0

    @property
    def elapsed(self) -> float:
        """Seconds recorded so far while recording, otherwise the playback position."""
        samples = self._recorded if self.recording else self.position
        return samples / self.SAMPLE_RATE

    def seek(self, seconds: float) -> None:
        if self.has_clip:
            self.position = int(min(max(seconds, 0.0), self.duration) * self.SAMPLE_RATE)

    def set_source(self, source: str) -> None:
        # Both buffers are sample-aligned, so switching mid-playback doesn't jump.
        if source not in SOURCES:
            raise ValueError(f"unknown preview source {source!r}")
        self.source = source

    # --- recording ----------------------------------------------------------------------------

    def start_recording(self) -> None:
        self.error = None
        if audio.sd is None:
            self.error = "sounddevice/PortAudio not available"
            return
        dev = devices.resolve("input", self.config.input_device)
        if dev is None:
            self.error = "No microphone found"
            return
        self.pause()
        self._chunks = []
        self._recorded = 0
        limit = int(self.MAX_SECONDS * self.SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
            take = min(frames, limit - self._recorded)
            self._chunks.append(np.array(mono[:take], dtype=np.float32))
            self._recorded += take
            if self._recorded >= limit:
                raise audio.sd.CallbackStop

        try:
            self._record_stream = audio.sd.InputStream(
                samplerate=self.SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=AudioPipeline.BLOCK_SIZE, device=dev.index, callback=callback,
                extra_settings=audio.wasapi_settings(dev.index),
            )
            self._record_stream.start()
            self.recording = True
        except Exception as e:
            self.stop_recording()
            self.error = f"Could not record from {dev.name}: {e}"
            log.error(self.error)

    @property
    def recording_finished(self) -> bool:
        """True once the stream hit MAX_SECONDS on its own and needs stop_recording()."""
        return self.recording and self._record_stream is not None and not self._record_stream.active

    def stop_recording(self) -> None:
        if self._record_stream is not None:
            try:
                self._record_stream.stop()
                self._record_stream.close()
            except Exception:
                log.exception("Error closing preview recording")
            self._record_stream = None
        if not self.recording:
            return
        self.recording = False
        if self._chunks:
            self.original = np.concatenate(self._chunks)
            self.processed = None
            self.position = 0
        self._chunks = []

    # --- processing ---------------------------------------------------------------------------

    def render(self) -> None:
        """Run the clip through the current effects. Blocking; call off the UI thread."""
        with self._render_lock:
            clip = self.original
            if clip is None:
                return
            if not self._models_loaded:
                self._pipeline._load_models()
                self._models_loaded = True
            self._pipeline.update_config(self.config)
            rendered = self._pipeline.render(clip)
            if clip is self.original:
                self.processed = rendered

    # --- playback -----------------------------------------------------------------------------

    def _buffer(self) -> np.ndarray | None:
        if self.source == "processed" and self.processed is not None:
            return self.processed
        return self.original

    def play(self) -> None:
        self.error = None
        if not self.has_clip or self.recording:
            return
        if audio.sd is None:
            self.error = "sounddevice/PortAudio not available"
            return
        self.pause()
        if self.position >= len(self.original):
            self.position = 0

        def callback(outdata, frames, time_info, status):
            clip = self._buffer()
            pos = self.position
            chunk = clip[pos:pos + frames] if clip is not None else np.zeros(0, np.float32)
            outdata.fill(0)
            outdata[:len(chunk)] = chunk[:, None]
            self.position = pos + len(chunk)
            if len(chunk) < frames:
                self.playing = False
                raise audio.sd.CallbackStop

        try:
            out_index = audio.sd.default.device[1]
            if out_index is None or out_index < 0:
                out_index = audio.sd.query_devices(kind="output")["index"]
            channels = min(2, audio.sd.query_devices(out_index)["max_output_channels"])
            self._play_stream = audio.sd.OutputStream(
                samplerate=self.SAMPLE_RATE, channels=channels, dtype="float32",
                blocksize=AudioPipeline.BLOCK_SIZE, device=out_index, callback=callback,
                extra_settings=audio.wasapi_settings(out_index),
            )
            self.playing = True
            self._play_stream.start()
        except Exception as e:
            self.pause()
            self.error = f"Could not play preview: {e}"
            log.error(self.error)

    def pause(self) -> None:
        self.playing = False
        if self._play_stream is not None:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception:
                log.exception("Error closing preview playback")
            self._play_stream = None

    def close(self) -> None:
        self.pause()
        self.stop_recording()
