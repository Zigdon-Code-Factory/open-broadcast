"""Shared styling and lightweight vector icons for the control panel."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

STYLESHEET = """
QWidget { color: #e7eeeb; font-size: 13px; }
QWidget#controlPanel { background: #111617; }
QWidget#tabPage, QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QScrollArea { border: none; }
QLabel { background: transparent; border: none; }
QLabel#brand { font-size: 19px; font-weight: 700; }
QLabel#pageTitle { font-size: 26px; font-weight: 600; }
QLabel#sectionTitle { font-size: 18px; font-weight: 600; }
QLabel#eyebrow { color: #a3b2b1; font-size: 11px; font-weight: 600; }
QLabel#muted { color: #a3b2b1; }
QLabel#helpText { color: #a3b2b1; font-size: 12px; }
QLabel#accent { color: #a8e675; font-weight: 600; }
QLabel#pipelineBadge {
    background: #20292a; color: #afbcbb; border: 1px solid #344043;
    border-radius: 14px; padding: 7px 16px; font-size: 11px; font-weight: 600;
}
QLabel#pipelineBadge[running="true"] {
    background: #203421; color: #a8e675; border-color: #395536;
}
QFrame#card, QGroupBox {
    background: #1b2224; border: 1px solid #303b3e; border-radius: 10px;
}
QGroupBox { font-weight: 600; margin-top: 10px; padding-top: 0; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 6px; }
QGroupBox:disabled { color: #859391; }
QLabel#preview {
    background: #0c1011; color: #8e9e9c; border: 1px solid #303b3e;
    border-radius: 8px; font-size: 15px;
}
QLabel#errorBox {
    background: #352426; color: #f2b7ad; border: 1px solid #69433e;
    border-radius: 8px; padding: 12px;
}
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: transparent; color: #a3b2b1; padding: 15px 24px;
    border-bottom: 3px solid transparent; margin-right: 10px;
    font-size: 14px; font-weight: 600;
}
QTabBar::tab:selected { color: #e7eeeb; border-bottom-color: #a8e675; }
QTabBar::tab:hover { background: #1b2224; color: #e7eeeb; }
QTabBar::tab:focus { color: #a8e675; }
QComboBox {
    background: #111719; border: 1px solid #435153; border-radius: 6px;
    padding: 8px 12px; min-height: 18px;
}
QComboBox:hover, QComboBox:focus { border-color: #a8e675; }
QComboBox:disabled { color: #859391; border-color: #303b3e; }
QComboBox QAbstractItemView {
    background: #1b2224; color: #e7eeeb; selection-background-color: #354630;
    selection-color: #ffffff; border: 1px solid #435153; padding: 4px;
}
QPushButton {
    background: #263033; border: 1px solid #435153; border-radius: 6px;
    padding: 9px 14px; font-weight: 600;
}
QPushButton:hover { background: #344043; border-color: #667a7a; }
QPushButton:pressed, QPushButton:checked { background: #3b4b46; }
QPushButton:focus { border-color: #a8e675; }
QPushButton:disabled { color: #859391; background: #20282a; border-color: #303b3e; }
QPushButton#segment { padding: 8px 12px; font-weight: 400; }
QPushButton#segment:checked {
    background: #203421; color: #a8e675; border-color: #395536; font-weight: 600;
}
QPushButton#primary {
    background: #a8e675; color: #14200e; border: 1px solid #a8e675; padding: 11px 24px;
}
QPushButton#primary:hover { background: #b9ee91; }
QPushButton#primary:disabled { background: #34462e; color: #a3b2b1; border-color: #43513d; }
QCheckBox { spacing: 9px; font-weight: 400; }
QCheckBox:disabled { color: #859391; }
QSlider::groove:horizontal { background: #435153; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #a8e675; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #e7eeeb; border: 1px solid #e7eeeb; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal:disabled { background: #435153; }
QSlider::handle:horizontal:disabled { background: #687774; border-color: #687774; }
QProgressBar {
    background: #0c1011; border: 1px solid #303b3e; border-radius: 6px;
}
QProgressBar::chunk { background: #a8e675; border-radius: 5px; }
QScrollBar:vertical { background: #111617; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #435153; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #263033; color: #e7eeeb; border: 1px solid #667a7a; padding: 6px; }
"""


def tab_icon(kind: str) -> QIcon:
    canvas = QPixmap(24, 24)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#b8c9c5"), 1.7, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    if kind == "microphone":
        painter.drawRoundedRect(QRectF(9, 3, 6, 11), 3, 3)
        outline = QPainterPath()
        outline.moveTo(6, 11)
        outline.cubicTo(6, 20, 18, 20, 18, 11)
        painter.drawPath(outline)
        painter.drawLine(12, 18, 12, 22)
        painter.drawLine(8, 22, 16, 22)
    elif kind == "speaker":
        outline = QPainterPath()
        outline.moveTo(3, 9)
        for point in ((7, 9), (12, 5), (12, 19), (7, 15), (3, 15)):
            outline.lineTo(*point)
        outline.closeSubpath()
        painter.drawPath(outline)
        waves = QPainterPath()
        waves.moveTo(16, 8)
        waves.cubicTo(20, 10, 20, 14, 16, 16)
        painter.drawPath(waves)
    elif kind == "camera":
        painter.drawRoundedRect(QRectF(2, 6, 13, 12), 2, 2)
        outline = QPainterPath()
        outline.moveTo(15, 10)
        for point in ((22, 6), (22, 18), (15, 14)):
            outline.lineTo(*point)
        painter.drawPath(outline)
    painter.end()
    return QIcon(canvas)
