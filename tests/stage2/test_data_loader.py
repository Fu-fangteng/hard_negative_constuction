from __future__ import annotations

import json
import pytest
import pandas as pd

from stage2.data_loader import NLIRecord, load_parquet, load_jsonl, save_preprocessed


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame({
        "anchor":   ["The cat sat on the mat.", "A dog ran fast."],
        "positive": ["A cat was sitting on a mat.", "The dog was running quickly."],
        "negative": ["The sky is blue.", "Fish swim in water."],
    })
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)
    return str(p)


@pytest.fixture
def sample_jsonl(tmp_path):
    lines = [
        {"anchor": "The cat sat.", "positive": "A cat was sitting.", "negative": "Sky is blue."},
        {"anchor": "A dog ran.", "positive": "The dog was running.", "negative": "Fish swim."},
    ]
    p = tmp_path / "test.jsonl"
    with p.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return str(p)


def test_load_parquet_returns_nli_records(sample_parquet):
    records = load_parquet(sample_parquet)
    assert len(records) == 2
    assert isinstance(records[0], NLIRecord)
    assert records[0].anchor == "The cat sat on the mat."
    assert records[0].pos == "A cat was sitting on a mat."
    assert records[0].neg == "The sky is blue."


def test_load_parquet_auto_generates_ids(sample_parquet):
    records = load_parquet(sample_parquet)
    assert records[0].id == "sample_000001"
    assert records[1].id == "sample_000002"


def test_load_parquet_sample_size(sample_parquet):
    records = load_parquet(sample_parquet, sample_size=1, seed=42)
    assert len(records) == 1


def test_load_parquet_normalizes_whitespace(tmp_path):
    df = pd.DataFrame({
        "anchor":   ["  hello   world  "],
        "positive": [" foo  bar "],
        "negative": ["baz"],
    })
    p = tmp_path / "ws.parquet"
    df.to_parquet(p, index=False)
    records = load_parquet(str(p))
    assert records[0].anchor == "hello world"
    assert records[0].pos == "foo bar"


def test_load_parquet_raises_on_missing_columns(tmp_path):
    df = pd.DataFrame({"anchor": ["text"], "text": ["other"]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_parquet(str(p))


def test_load_jsonl_returns_nli_records(sample_jsonl):
    records = load_jsonl(sample_jsonl)
    assert len(records) == 2
    assert isinstance(records[0], NLIRecord)
    assert records[0].anchor == "The cat sat."
    assert records[0].pos == "A cat was sitting."
    assert records[0].neg == "Sky is blue."


def test_load_jsonl_sample_size(sample_jsonl):
    records = load_jsonl(sample_jsonl, sample_size=1, seed=42)
    assert len(records) == 1


def test_save_preprocessed_writes_json(sample_parquet, tmp_path):
    records = load_parquet(sample_parquet)
    out = tmp_path / "preprocessed_data.json"
    save_preprocessed(records, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert data[0] == {
        "id": "sample_000001",
        "anchor": "The cat sat on the mat.",
        "pos": "A cat was sitting on a mat.",
        "neg": "The sky is blue.",
    }
