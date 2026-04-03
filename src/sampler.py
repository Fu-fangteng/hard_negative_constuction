from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

from .data_utils import STSRecord, export_records, load_data


def _validate_records(records: Sequence[STSRecord]) -> None:
    if not records:
        raise ValueError("Input records are empty")
    for idx, record in enumerate(records, start=1):
        if not record.text1 or not record.text2:
            raise ValueError(f"Invalid record at position {idx}: empty text fields")


def select_top_k_by_score(records: Sequence[STSRecord], k: int = 100) -> List[STSRecord]:
    """
    Select top-k high-similarity positive pairs for hard-negative construction.
    Tie-breaker: score desc, id asc.
    """
    _validate_records(records)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(records):
        k = len(records)

    ranked = sorted(records, key=lambda r: (-r.score, r.id))
    return ranked[:k]


def random_sample(
    records: Sequence[STSRecord],
    sample_size: int,
    seed: int = 42,
) -> List[STSRecord]:
    """
    Optional helper for ablation/debug datasets with deterministic randomness.
    """
    _validate_records(records)
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size > len(records):
        raise ValueError("sample_size cannot exceed record count")

    rng = random.Random(seed)
    return rng.sample(list(records), sample_size)


def build_top_k_dataset(
    input_path: str | Path,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    k: int = 100,
) -> List[STSRecord]:
    """
    Load raw data -> normalize schema -> select top-k by score -> optional export.
    """
    records = load_data(input_path)
    top_k = select_top_k_by_score(records, k=k+1)
    if output_json_path is not None or output_csv_path is not None:
        export_records(top_k, json_path=output_json_path, csv_path=output_csv_path)
    return top_k
