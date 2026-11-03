"""Headless self-test used by CI to verify a frozen build has every model and native library.

    OpenBroadcast --self-test [--self-test-output result.txt]

Windowed builds have no stdout, so the result is also written to a file when requested.
Exit code 0 means every check passed.
"""

from __future__ import annotations

import platform
import traceback
from collections.abc import Callable

import numpy as np

from open_broadcast import __version__


def _audio() -> str:
    from open_broadcast.pipeline.audio import DeepFilterNet, SileroVAD

    dfn = DeepFilterNet()
    out = dfn.process(np.zeros(DeepFilterNet.HOP_SIZE * 4, dtype=np.float32))
    assert out.shape == (DeepFilterNet.HOP_SIZE * 4,), out.shape
    p = SileroVAD().probability(np.zeros(SileroVAD.WINDOW, dtype=np.float32))
    assert 0.0 <= p <= 1.0, p
    return "deepfilternet + silero ok"


def _video() -> str:
    from open_broadcast.pipeline.video import FaceDetector, Segmenter

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    seg = Segmenter()
    mask = seg.mask(rgb, 0)
    seg.close()
    det = FaceDetector()
    det.center(rgb, 0)
    det.close()
    return f"segmenter mask {mask.shape} + face detector ok"


def _libs() -> str:
    import cv2
    import sounddevice  # noqa: F401  (loads PortAudio)
    from PySide6.QtWidgets import QApplication  # noqa: F401

    return f"cv2 {cv2.__version__}, sounddevice, PySide6 ok"


def run(output: str | None = None) -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("libs", _libs), ("audio", _audio), ("video", _video),
    ]
    lines = [f"Open Broadcast {__version__} self-test on {platform.system()} {platform.machine()}"]
    failed = False
    for name, fn in checks:
        try:
            lines.append(f"PASS {name}: {fn()}")
        except Exception:
            failed = True
            lines.append(f"FAIL {name}:\n{traceback.format_exc()}")
    lines.append("FAILED" if failed else "OK")
    text = "\n".join(lines) + "\n"
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    try:
        print(text, end="")
    except Exception:  # no stdout in windowed builds
        pass
    return 1 if failed else 0
