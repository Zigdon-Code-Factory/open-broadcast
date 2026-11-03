import json

from open_broadcast import config


def test_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    cfg = config.PipelineConfig(camera_index=2, output_device="CABLE Input", vad_threshold=0.3)
    config.save(cfg, p)
    assert config.load(p) == cfg


def test_unknown_keys_and_bad_values(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"bogus": 1, "background_mode": "sparkles", "auto_frame": True}))
    cfg = config.load(p)
    assert cfg.background_mode == "blur"
    assert cfg.auto_frame is True


def test_corrupt_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    assert config.load(p) == config.PipelineConfig()
