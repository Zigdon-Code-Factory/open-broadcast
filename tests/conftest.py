import os
from pathlib import Path

import pytest
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from open_broadcast import models


@pytest.fixture(scope="session")
def application():
    instance = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "")) / "Fonts" / "segoeui.ttf"
    if not QFontDatabase.families() and font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            instance.setFont(QFont(families[0], 10))
    return instance


def _have_model(name: str) -> bool:
    try:
        models.model_path(name, allow_download=False)
        return True
    except models.ModelUnavailableError:
        return False


@pytest.fixture
def require_model():
    def _require(name: str):
        if not _have_model(name):
            pytest.skip(f"model {name} not available (run scripts/fetch_models.py)")
    return _require
