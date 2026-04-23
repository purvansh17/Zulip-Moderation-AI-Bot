from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.ingest_and_expand import ingest_csv


def test_ingest_csv_triggers_initial_snapshot(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("text,is_suicide,is_toxicity\nhello,0,0\n", encoding="utf-8")

    chunk = pd.DataFrame([{"text": "hello", "is_suicide": 0, "is_toxicity": 0}])
    minio = MagicMock()
    minio.bucket_exists.return_value = True

    with (
        patch("src.data.ingest_and_expand.get_minio_client", return_value=minio),
        patch("src.data.ingest_and_expand.pd.read_csv", return_value=iter([chunk])),
        patch("src.data.ingest_and_expand.run_training_snapshot") as mock_trigger,
    ):
        chunk_count = ingest_csv(csv_path=str(csv_path), bucket="proj09_Data")

    assert chunk_count == 1
    minio.put_object.assert_called_once()
    mock_trigger.assert_called_once()
    assert mock_trigger.call_args.args[0] == "initial"
