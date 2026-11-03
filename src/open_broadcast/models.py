"""Model registry: pinned download URLs + SHA-256 checksums.

Release builds ship every model inside the bundle (see scripts/fetch_models.py).
Source installs download them on first use into the per-user cache directory.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from open_broadcast import paths

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str
    # When the URL points at a tar archive, the member to extract.
    tar_member: str | None = None


MODELS: dict[str, ModelSpec] = {
    # DeepFilterNet3 streaming model (torchDF port, hop=512 @ 48 kHz). MIT/Apache-2.0.
    "deepfilternet": ModelSpec(
        filename="deepfilternet3_streaming.onnx",
        url=(
            "https://github.com/grazder/DeepFilterNet/raw/"
            "408414f2bdb9ed772580558d1c3ec26c5dc002d7/models/DeepFilterNet3_torchDF_onnx.tar"
        ),
        sha256="b758c49d6708a5b7979e3de185705a8a4915076c862fb17b1b304d9a72b75cdc",
        tar_member="DeepFilterNet3_torchDF_onnx/denoiser_model.onnx",
    ),
    # Silero VAD v6. MIT.
    "silero_vad": ModelSpec(
        filename="silero_vad.onnx",
        url="https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    ),
    # MediaPipe selfie segmenter. Apache-2.0.
    "selfie_segmenter": ModelSpec(
        filename="selfie_segmenter.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_segmenter/float16/1/selfie_segmenter.tflite"
        ),
        sha256="191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
    ),
    # MediaPipe BlazeFace short range. Apache-2.0.
    "face_detector": ModelSpec(
        filename="blaze_face_short_range.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
        sha256="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
}


class ModelUnavailableError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(spec: ModelSpec, dest: Path) -> Path:
    """Download (and extract, if needed) a model to ``dest``, verifying its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from %s", spec.filename, spec.url)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "open-broadcast"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()

    if spec.tar_member:
        with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
            member = tar.extractfile(spec.tar_member)
            if member is None:
                raise ModelUnavailableError(f"{spec.tar_member} missing from archive")
            payload = member.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec.sha256:
        raise ModelUnavailableError(
            f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, got {digest}"
        )

    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp.write(payload)
    Path(tmp.name).replace(dest)
    return dest


def model_path(name: str, allow_download: bool = True) -> Path:
    """Return a local path for model ``name``, downloading it if necessary."""
    spec = MODELS[name]
    bundled = paths.bundled_models_dir() / spec.filename
    if bundled.is_file():
        return bundled

    cached = paths.cache_dir() / "models" / spec.filename
    if cached.is_file() and _sha256(cached) == spec.sha256:
        return cached

    if not allow_download:
        raise ModelUnavailableError(f"Model {name} not found and downloads are disabled")
    try:
        return download(spec, cached)
    except ModelUnavailableError:
        raise
    except Exception as e:  # network errors etc.
        raise ModelUnavailableError(f"Could not download {name}: {e}") from e
