"""Render the app icon to PNG + ICO. Run once; outputs are committed.

    python scripts/make_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPen

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "src" / "open_broadcast" / "assets" / "icon.png"
ICO = ROOT / "packaging" / "icon.ico"
SIZE = 1024


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 1024

    grad = QLinearGradient(QPointF(0, 0), QPointF(size, size))
    grad.setColorAt(0, QColor("#22c55e"))
    grad.setColorAt(1, QColor("#0e7490"))
    p.setBrush(grad)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(48 * s, 48 * s, 928 * s, 928 * s), 200 * s, 200 * s)

    # Center dot + two broadcast arcs on each side.
    white = QColor("white")
    p.setBrush(white)
    c = QPointF(512 * s, 512 * s)
    p.drawEllipse(c, 90 * s, 90 * s)
    pen = QPen(white, 64 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for r in (220, 340):
        rect = QRectF(c.x() - r * s, c.y() - r * s, 2 * r * s, 2 * r * s)
        p.drawArc(rect, int(-40 * 16), int(80 * 16))
        p.drawArc(rect, int(140 * 16), int(80 * 16))
    p.end()
    return img


def main() -> int:
    QGuiApplication(sys.argv)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    ICO.parent.mkdir(parents=True, exist_ok=True)
    img = render(SIZE)
    img.save(str(PNG))
    Image.open(PNG).save(ICO, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                     (128, 128), (256, 256)])
    print(f"Wrote {PNG} and {ICO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
