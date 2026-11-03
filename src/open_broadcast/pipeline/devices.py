"""Audio device discovery (name-based, so selections survive device re-enumeration)."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
except (ImportError, OSError):  # OSError: PortAudio library missing
    sd = None

# Substrings identifying virtual-mic sinks, in order of preference.
VIRTUAL_OUTPUT_HINTS = ("CABLE Input", "OpenBroadcast", "BlackHole", "Loopback")


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int


def preferred_hostapi() -> int | None:
    """WASAPI on Windows (low latency + auto sample-rate conversion), default elsewhere."""
    if sd is None:
        return None
    if platform.system() == "Windows":
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in api["name"]:
                return i
    return sd.default.hostapi if sd.default.hostapi >= 0 else None


def list_devices(kind: str) -> list[AudioDevice]:
    """List ``"input"`` or ``"output"`` devices on the preferred host API."""
    if sd is None:
        return []
    api = preferred_hostapi()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    out = []
    for i, d in enumerate(sd.query_devices()):
        if api is not None and d["hostapi"] != api:
            continue
        if d[key] > 0:
            out.append(AudioDevice(i, d["name"], d["max_input_channels"], d["max_output_channels"]))
    return out


def default_device(kind: str) -> AudioDevice | None:
    devices = list_devices(kind)
    if not devices:
        return None
    if kind == "output":
        for hint in VIRTUAL_OUTPUT_HINTS:
            for d in devices:
                if hint.lower() in d.name.lower():
                    return d
    api = preferred_hostapi()
    if api is not None:
        idx = sd.query_hostapis(api)["default_input_device" if kind == "input" else
                                     "default_output_device"]
        for d in devices:
            if d.index == idx:
                return d
    return devices[0]


def resolve(kind: str, name: str | None) -> AudioDevice | None:
    """Find a device by saved name, falling back to the default."""
    if name:
        for d in list_devices(kind):
            if d.name == name:
                return d
        log.warning("Saved %s device %r not found; using default", kind, name)
    return default_device(kind)
