"""
Stage 2 Data Loader
===================
加载 nli_for_simcse 格式数据（parquet 或 jsonl），
统一输出 NLIRecord(id, anchor, pos, neg) 列表。
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


@dataclass
class NLIRecord:
    id: str
    anchor: str
    pos: str
    neg: str


def load_parquet(
    path: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[NLIRecord]:
    df = pd.read_parquet(path)
    required = {"anchor", "positive", "negative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file missing required columns: {missing}")
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    records: List[NLIRecord] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        records.append(NLIRecord(
            id=f"sample_{idx:06d}",
            anchor=_normalize(getattr(row, "anchor", "")),
            pos=_normalize(getattr(row, "positive", "")),
            neg=_normalize(getattr(row, "negative", "")),
        ))
    return records


def load_jsonl(
    path: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[NLIRecord]:
    """加载 nli_for_simcse 格式的 jsonl 文件（字段 anchor/positive/negative）。"""
    raw_lines: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(json.loads(line))
    if sample_size is not None and sample_size < len(raw_lines):
        rng = random.Random(seed)
        raw_lines = rng.sample(raw_lines, sample_size)
    records: List[NLIRecord] = []
    for idx, raw in enumerate(raw_lines, start=1):
        records.append(NLIRecord(
            id=str(raw.get("id", f"sample_{idx:06d}")),
            anchor=_normalize(raw.get("anchor", raw.get("sentence1", ""))),
            pos=_normalize(raw.get("positive", raw.get("pos", ""))),
            neg=_normalize(raw.get("negative", raw.get("neg", ""))),
        ))
    return records


def load_data(
    path: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[NLIRecord]:
    """根据文件后缀自动选择加载方式（.parquet / .jsonl）。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return load_parquet(path, sample_size=sample_size, seed=seed)
    if suffix == ".jsonl":
        return load_jsonl(path, sample_size=sample_size, seed=seed)
    raise ValueError(f"Unsupported file format: {suffix!r}. Use .parquet or .jsonl")


def save_preprocessed(records: List[NLIRecord], output_path: str) -> None:
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
