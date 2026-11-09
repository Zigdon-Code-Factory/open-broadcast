"""User configuration, persisted as JSON in the per-user config directory."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from open_broadcast import paths

log = logging.getLogger(__name__)

BACKGROUND_MODES = ("blur", "remove", "replace", "off")
# Noise-reduction limits at or above this are treated as "no limit" (full DeepFilterNet).
NOISE_REDUCTION_MAX_DB = 100.0


@dataclass
class PipelineConfig:
    # Video
    video_enabled: bool = True
    camera_index: int = 0
    background_mode: str = "blur"  # one of BACKGROUND_MODES
    blur_strength: int = 21
    background_image: str | None = None
    auto_frame: bool = False
    auto_frame_zoom: float = 1.4
    auto_frame_smoothing: float = 0.85

    # Audio
    audio_enabled: bool = True
    noise_suppression: bool = True
    # Max noise attenuation in dB (DeepFilterNet "attenuation limit"). Lower keeps more of
    # the room, which sounds more natural; NOISE_REDUCTION_MAX_DB or above means unlimited.
    noise_reduction_db: float = 100.0
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    # Seconds the gate stays open after speech stops, so word endings aren't clipped.
    vad_holdover: float = 0.3
    # High-pass filter for desk thumps, HVAC rumble, and mic-handling noise.
    low_cut: bool = False
    low_cut_hz: float = 80.0
    # Device *names* rather than indices: indices shift when devices are plugged in.
    input_device: str | None = None
    output_device: str | None = None

    # App
    check_for_updates: bool = True
    skipped_version: str | None = None


def default_path() -> Path:
    return paths.config_dir() / "config.json"


def load(path: Path | None = None) -> PipelineConfig:
    path = path or default_path()
    cfg = PipelineConfig()
    if not path.is_file():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("Ignoring unreadable config %s: %s", path, e)
        return cfg
    known = {f.name: f for f in fields(PipelineConfig)}
    for key, value in data.items():
        if key in known:
            setattr(cfg, key, value)
    if cfg.background_mode not in BACKGROUND_MODES:
        cfg.background_mode = "blur"
    try:
        cfg.noise_reduction_db = min(max(float(cfg.noise_reduction_db), 0.0),
                                     NOISE_REDUCTION_MAX_DB)
        cfg.vad_holdover = min(max(float(cfg.vad_holdover), 0.05), 2.0)
        cfg.low_cut_hz = min(max(float(cfg.low_cut_hz), 20.0), 300.0)
    except (TypeError, ValueError):
        log.warning("Ignoring invalid audio tuning values in %s", path)
        defaults = PipelineConfig()
        cfg.noise_reduction_db = defaults.noise_reduction_db
        cfg.vad_holdover = defaults.vad_holdover
        cfg.low_cut_hz = defaults.low_cut_hz
    return cfg


def save(cfg: PipelineConfig, path: Path | None = None) -> None:
    path = path or default_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("Could not save config to %s: %s", path, e)
