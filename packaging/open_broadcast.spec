# PyInstaller spec for Open Broadcast (Windows / macOS / Linux).
#
#   python scripts/fetch_models.py
#   pyinstaller packaging/open_broadcast.spec --noconfirm
#
# Produces dist/OpenBroadcast/ (onedir) and, on macOS, dist/OpenBroadcast.app.
# ruff: noqa
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
PKG = SRC / "open_broadcast"

sys.path.insert(0, str(SRC))
from open_broadcast import __version__  # noqa: E402

models_dir = PKG / "models"
if not any(models_dir.glob("*.onnx")):
    raise SystemExit("Models missing: run `python scripts/fetch_models.py` first")

datas = [
    (str(PKG / "assets"), "open_broadcast/assets"),
    (str(models_dir), "open_broadcast/models"),
]
datas += collect_data_files("mediapipe", excludes=["**/*.pyc", "**/testdata/**", "**/examples/**"])
binaries = collect_dynamic_libs("mediapipe") + collect_dynamic_libs("onnxruntime")
hiddenimports = collect_submodules("mediapipe.tasks.python") + ["pyvirtualcam"]
if sys.platform == "win32":
    hiddenimports += ["pygrabber.dshow_graph", "comtypes.stream"]

a = Analysis(
    [str(PKG / "__main__.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "torch", "torchaudio", "tensorflow", "IPython", "pytest",
              "PySide6.Qt3DCore", "PySide6.QtWebEngineCore", "PySide6.QtQuick", "PySide6.QtQml",
              "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization",
              "PySide6.QtMultimedia", "PySide6.QtBluetooth", "PySide6.QtSql"],
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = None
if sys.platform == "win32":
    icon = str(ROOT / "packaging" / "icon.ico")
elif sys.platform == "darwin":
    icon = str(PKG / "assets" / "icon.png")  # PyInstaller converts via Pillow

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OpenBroadcast",
    console=False,
    icon=icon,
    version=str(ROOT / "packaging" / "windows" / "version_info.txt")
    if sys.platform == "win32" and (ROOT / "packaging" / "windows" / "version_info.txt").exists()
    else None,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="OpenBroadcast", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OpenBroadcast.app",
        icon=icon,
        bundle_identifier="io.github.zigdon.openbroadcast",
        version=__version__,
        info_plist={
            "CFBundleName": "Open Broadcast",
            "CFBundleDisplayName": "Open Broadcast",
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "LSUIElement": True,  # tray app: no Dock icon
            "LSMinimumSystemVersion": "12.0",
            "NSCameraUsageDescription": "Open Broadcast processes your webcam video.",
            "NSMicrophoneUsageDescription": "Open Broadcast removes noise from your microphone.",
            "NSHighResolutionCapable": True,
        },
    )
