"""Tests for YAML-backed Config loading."""


import pytest

from src.utils.config import Config, _env_kwargs, _load_yaml_defaults, _yaml_to_kwargs


def test_yaml_loads(tmp_path):
    """Config loads values from YAML file."""
    yaml_content = b"ingestion:\n  chunk_size: 99999\n"
    yaml_file = tmp_path / "pipeline.yaml"
    yaml_file.write_bytes(yaml_content)

    defaults = _load_yaml_defaults(str(yaml_file))
    kwargs = _yaml_to_kwargs(defaults)
    cfg = Config(**kwargs)

    assert cfg.CHUNK_SIZE == 99999


def test_missing_file_fallback():
    """Config uses defaults when YAML file is missing."""
    defaults = _load_yaml_defaults("/nonexistent/path/pipeline.yaml")
    assert defaults == {}
    cfg = Config()
    assert cfg.CHUNK_SIZE == 50_000  # default value


def test_yaml_to_kwargs_mapping():
    """_yaml_to_kwargs maps YAML section.key to Config field names."""
    data = {
        "ingestion": {"chunk_size": 100},
        "quality": {"min_text_length": 5},
        "traffic": {"rps_target": 20},
    }
    result = _yaml_to_kwargs(data)
    assert result == {
        "CHUNK_SIZE": 100,
        "QUALITY_MIN_TEXT_LENGTH": 5,
        "RPS_TARGET": 20,
    }


def test_env_var_override(monkeypatch):
    """Env vars take precedence over YAML defaults."""
    monkeypatch.setenv("MINIO_ENDPOINT", "remote:9999")
    env = _env_kwargs()
    cfg = Config(**env)
    assert cfg.MINIO_ENDPOINT == "remote:9999"


def test_config_is_frozen():
    """Config dataclass is immutable."""
    cfg = Config()
    with pytest.raises(AttributeError):
        cfg.CHUNK_SIZE = 123


def test_all_tunable_params_present(tmp_path):
    """YAML with all params creates Config with expected values."""
    yaml_content = (
        b"ingestion:\n"
        b"  chunk_size: 25000\n"
        b"  synthetic_target_rows: 5000\n"
        b"quality:\n"
        b"  min_text_length: 5\n"
        b"  max_text_length: 3000\n"
        b"  error_pattern: '#ERROR!'\n"
        b"split:\n"
        b"  train_ratio: 0.80\n"
        b"  val_ratio: 0.10\n"
        b"  test_ratio: 0.10\n"
        b"  random_state: 123\n"
        b"buckets:\n"
        b"  raw: my-raw-bucket\n"
        b"  training: my-training-bucket\n"
        b"traffic:\n"
        b"  rps_target: 20\n"
        b"batch:\n"
        b"  upload_size: 5000\n"
    )
    yaml_file = tmp_path / "pipeline.yaml"
    yaml_file.write_bytes(yaml_content)

    defaults = _load_yaml_defaults(str(yaml_file))
    kwargs = _yaml_to_kwargs(defaults)
    cfg = Config(**kwargs)

    assert cfg.CHUNK_SIZE == 25000
    assert cfg.SYNTHETIC_TARGET_ROWS == 5000
    assert cfg.QUALITY_MIN_TEXT_LENGTH == 5
    assert cfg.QUALITY_MAX_TEXT_LENGTH == 3000
    assert cfg.QUALITY_ERROR_PATTERN == "#ERROR!"
    assert cfg.TRAIN_SPLIT_RATIO == 0.80
    assert cfg.VAL_SPLIT_RATIO == 0.10
    assert cfg.TEST_SPLIT_RATIO == 0.10
    assert cfg.RANDOM_STATE == 123
    assert cfg.BUCKET_RAW == "my-raw-bucket"
    assert cfg.BUCKET_TRAINING == "my-training-bucket"
    assert cfg.RPS_TARGET == 20
    assert cfg.MINIO_BATCH_UPLOAD_SIZE == 5000
