"""Video pipeline: webcam -> background blur/removal + auto-frame -> virtual camera."""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from open_broadcast import models

if TYPE_CHECKING:
    from open_broadcast.config import PipelineConfig

log = logging.getLogger(__name__)

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


VCAM_HINTS = {
    "Windows": "Install OBS Studio (it provides 'OBS Virtual Camera'), then restart.",
    "Darwin": "Install OBS Studio and start its virtual camera once, then restart.",
    "Linux": "Load v4l2loopback: sudo modprobe v4l2loopback devices=1 exclusive_caps=1",
}


def list_cameras(max_probe: int = 5) -> list[tuple[int, str]]:
    """(index, name) pairs for available cameras. Names come from the OS where possible."""
    system = platform.system()
    if system == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph

            # DirectShow enumeration order matches cv2.CAP_DSHOW indices.
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception as e:
            log.debug("DirectShow enumeration failed: %s", e)
    elif system == "Linux":
        from pathlib import Path

        cams = []
        for dev in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                idx = int(dev.name.removeprefix("video"))
                cams.append((idx, (dev / "name").read_text().strip()))
            except (ValueError, OSError):
                continue
        if cams:
            return cams
    return [(i, f"Camera {i}") for i in range(max_probe)]


def open_camera(index: int, width: int = 1280, height: int = 720, fps: int = 30):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            return cap
        cap.release()
    return None


class Segmenter:
    """MediaPipe selfie segmenter (Tasks API). Returns a float32 person mask in [0, 1]."""

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(models.model_path("selfie_segmenter"))),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def mask(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        import mediapipe as mp

        result = self._seg.segment_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        return result.confidence_masks[-1].numpy_view().squeeze()

    def close(self):
        self._seg.close()


class FaceDetector:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(models.model_path("face_detector"))),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.6,
        )
        self._det = vision.FaceDetector.create_from_options(opts)

    def center(self, rgb: np.ndarray, ts_ms: int) -> tuple[float, float] | None:
        """Normalized (x, y) center of the most confident face, or None."""
        import mediapipe as mp

        result = self._det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.detections:
            return None
        best = max(result.detections, key=lambda d: d.categories[0].score)
        box = best.bounding_box
        h, w = rgb.shape[:2]
        return ((box.origin_x + box.width / 2) / w, (box.origin_y + box.height / 2) / h)

    def close(self):
        self._det.close()


def composite(frame: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Alpha-blend ``frame`` over ``background`` using ``mask`` (person = 1)."""
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)[..., None]
    out = background.astype(np.float32) + (frame.astype(np.float32) - background) * alpha
    return out.clip(0, 255).astype(np.uint8)


def fast_blur(frame: np.ndarray, strength: int) -> np.ndarray:
    """Large-kernel blur computed at quarter resolution for speed."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(1, strength // 4) * 2 + 1
    small = cv2.GaussianBlur(small, (k, k), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_to_center(frame: np.ndarray, center: tuple[float, float], zoom: float) -> np.ndarray:
    h, w = frame.shape[:2]
    crop_w, crop_h = int(w / zoom), int(h / zoom)
    cx, cy = int(center[0] * w), int(center[1] * h)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    return cv2.resize(frame[y1:y1 + crop_h, x1:x1 + crop_w], (w, h))


class VideoPipeline:
    """Processes webcam frames on a worker thread."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._smoothed_center: np.ndarray | None = None
        self._bg_cache: tuple[str, tuple[int, int], np.ndarray | None] | None = None
        self.latest_frame: np.ndarray | None = None
        self.fps = 0.0
        self.error: str | None = None
        self.vcam_device: str | None = None

    def start(self) -> None:
        self.error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self.latest_frame = None

    def update_config(self, config: PipelineConfig) -> None:
        self.config = config

    def _background(self, frame: np.ndarray) -> np.ndarray | None:
        cfg = self.config
        if cfg.background_mode == "blur":
            return fast_blur(frame, cfg.blur_strength)
        if cfg.background_mode == "remove":
            return np.zeros_like(frame)
        if cfg.background_mode == "replace":
            size = (frame.shape[1], frame.shape[0])
            path = cfg.background_image or ""
            if not self._bg_cache or self._bg_cache[:2] != (path, size):
                img = cv2.imread(path) if path else None
                img = cv2.resize(img, size) if img is not None else None
                self._bg_cache = (path, size, img)
            img = self._bg_cache[2]
            return img if img is not None else np.zeros_like(frame)
        return None

    def process_frame(self, frame, segmenter, face_detector, ts_ms: int) -> np.ndarray:
        cfg = self.config
        rgb = None
        if cfg.background_mode != "off" and segmenter is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bg = self._background(frame)
            if bg is not None:
                frame = composite(frame, segmenter.mask(rgb, ts_ms), bg)

        if cfg.auto_frame and face_detector is not None:
            if rgb is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            c = face_detector.center(rgb, ts_ms)
            if c is not None:
                target = np.array(c)
                if self._smoothed_center is None:
                    self._smoothed_center = target
                else:
                    s = cfg.auto_frame_smoothing
                    self._smoothed_center = s * self._smoothed_center + (1 - s) * target
            if self._smoothed_center is not None:
                frame = crop_to_center(frame, tuple(self._smoothed_center), cfg.auto_frame_zoom)
        else:
            self._smoothed_center = None
        return frame

    def _run(self) -> None:
        segmenter = face_detector = None
        try:
            segmenter = Segmenter()
        except Exception as e:
            log.error("Segmentation unavailable: %s", e)
        try:
            face_detector = FaceDetector()
        except Exception as e:
            log.error("Face detection unavailable: %s", e)

        cap = open_camera(self.config.camera_index)
        if cap is None:
            self.error = f"Could not open camera {self.config.camera_index}"
            log.error(self.error)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        log.info("Camera %d opened: %dx%d @ %d fps", self.config.camera_index, width, height, fps)

        vcam = None
        if pyvirtualcam is not None:
            try:
                vcam = pyvirtualcam.Camera(
                    width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
                )
                self.vcam_device = vcam.device
                log.info("Virtual camera: %s", vcam.device)
            except Exception as e:
                log.warning("Virtual camera unavailable: %s", e)
                self.error = "Virtual camera not found. " + VCAM_HINTS.get(
                    platform.system(), "Install a virtual camera driver."
                )
        else:
            self.error = "pyvirtualcam not installed"

        last_ts = 0
        frames, t_window = 0, time.monotonic()
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                ts = max(last_ts + 1, int(time.monotonic() * 1000))
                last_ts = ts
                try:
                    frame = self.process_frame(frame, segmenter, face_detector, ts)
                except Exception:
                    log.exception("Frame processing failed")
                self.latest_frame = frame
                if vcam is not None:
                    vcam.send(frame)

                frames += 1
                now = time.monotonic()
                if now - t_window >= 1.0:
                    self.fps = frames / (now - t_window)
                    frames, t_window = 0, now
        finally:
            cap.release()
            if vcam is not None:
                vcam.close()
            if segmenter:
                segmenter.close()
            if face_detector:
                face_detector.close()
            self.fps = 0.0
