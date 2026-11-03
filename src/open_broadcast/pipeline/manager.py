"""Pipeline manager: owns the video + audio pipelines and the shared config."""

from __future__ import annotations

import logging
import threading

from open_broadcast import config as config_mod
from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline.audio import AudioPipeline
from open_broadcast.pipeline.video import VideoPipeline

log = logging.getLogger(__name__)

# Changing these requires reopening devices.
RESTART_KEYS = {"camera_index", "input_device", "output_device", "video_enabled", "audio_enabled"}


class PipelineManager:
    def __init__(self, config: PipelineConfig | None = None, persist: bool = True):
        self.config = config or PipelineConfig()
        self._persist = persist
        self._video: VideoPipeline | None = None
        self._audio: AudioPipeline | None = None
        self._running = False
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def video(self) -> VideoPipeline | None:
        return self._video

    @property
    def audio(self) -> AudioPipeline | None:
        return self._audio

    def errors(self) -> list[str]:
        return [p.error for p in (self._video, self._audio) if p is not None and p.error]

    def start(self) -> None:
        """Start pipelines. May block while models load/download; call off the UI thread."""
        with self._lock:
            if self._running:
                return
            if self.config.video_enabled:
                self._video = VideoPipeline(self.config)
                self._video.start()
            if self.config.audio_enabled:
                self._audio = AudioPipeline(self.config)
                self._audio.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            if self._video:
                self._video.stop()
            if self._audio:
                self._audio.stop()
            self._video = self._audio = None
            self._running = False

    def update_config(self, **kwargs) -> bool:
        """Apply settings. Returns True if a restart is needed for them to take effect."""
        changed = set()
        for k, v in kwargs.items():
            if hasattr(self.config, k) and getattr(self.config, k) != v:
                setattr(self.config, k, v)
                changed.add(k)
        if not changed:
            return False
        if self._persist:
            config_mod.save(self.config)
        for p in (self._video, self._audio):
            if p is not None:
                p.update_config(self.config)
        return self._running and bool(changed & RESTART_KEYS)
