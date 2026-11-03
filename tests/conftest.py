import pytest

from open_broadcast import models


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
