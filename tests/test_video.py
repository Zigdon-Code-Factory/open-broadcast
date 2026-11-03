import numpy as np

from open_broadcast.config import PipelineConfig
from open_broadcast.pipeline.video import (
    FaceDetector,
    Segmenter,
    VideoPipeline,
    composite,
    crop_to_center,
    fast_blur,
)


def test_composite_uses_mask():
    fg = np.full((40, 40, 3), 200, np.uint8)
    bg = np.zeros_like(fg)
    mask = np.zeros((40, 40), np.float32)
    mask[:, :20] = 1.0
    out = composite(fg, mask, bg)
    assert out[20, 2].mean() > 190 and out[20, 37].mean() < 10


def test_crop_keeps_size_and_clamps():
    frame = np.random.randint(0, 255, (72, 128, 3), np.uint8)
    out = crop_to_center(frame, (0.0, 0.0), 2.0)
    assert out.shape == frame.shape


def test_fast_blur_shape():
    frame = np.random.randint(0, 255, (72, 128, 3), np.uint8)
    assert fast_blur(frame, 21).shape == frame.shape


def test_process_frame_all_modes(require_model):
    require_model("selfie_segmenter")
    require_model("face_detector")
    seg, face = Segmenter(), FaceDetector()
    try:
        frame = np.random.randint(0, 255, (360, 640, 3), np.uint8)
        for i, mode in enumerate(["blur", "remove", "replace", "off"]):
            cfg = PipelineConfig(background_mode=mode, auto_frame=True)
            out = VideoPipeline(cfg).process_frame(frame, seg, face, ts_ms=i + 1)
            assert out.shape == frame.shape and out.dtype == np.uint8
    finally:
        seg.close()
        face.close()
