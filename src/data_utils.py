from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class STSRecord:
    """Canonical schema used across the full pipeline."""

    id: str
    text1: str
    text2: str
    score: float


def normalize_text(text: Optional[str]) -> str:
    """Trim and collapse whitespace to avoid noisy token differences."""
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_record(raw: Dict[str, Any], index: int) -> STSRecord:
    """
    Parse one raw record into canonical STS schema:
    id, text1, text2, score.
    """
    record_id = str(raw.get("id") or f"sample_{index:06d}")
    text1 = normalize_text(raw.get("text1", raw.get("sentence1")))
    text2 = normalize_text(raw.get("text2", raw.get("sentence2")))
    score = _to_float(raw.get("score"), default=0.0)

    if not text1:
        raise ValueError(f"record {record_id}: text1 is empty")
    if not text2:
        raise ValueError(f"record {record_id}: text2 is empty")

    return STSRecord(id=record_id, text1=text1, text2=text2, score=score)


def load_jsonl(path: str | Path) -> List[STSRecord]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    records: List[STSRecord] = []
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid jsonl at line {idx}: {exc}") from exc
            records.append(parse_record(raw, idx))
    return records


def load_json(path: str | Path) -> List[STSRecord]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    with src.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list of objects")

    return [parse_record(item, idx) for idx, item in enumerate(payload, start=1)]


def load_csv(path: str | Path) -> List[STSRecord]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    records: List[STSRecord] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            records.append(parse_record(row, idx))
    return records


def load_data(path: str | Path) -> List[STSRecord]:
    """
    Unified loader by file suffix.
    Supported: .jsonl, .json, .csv
    """
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".jsonl":
        records = load_jsonl(src)
    elif suffix == ".json":
        records = load_json(src)
    elif suffix == ".csv":
        records = load_csv(src)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if not records:
        raise ValueError(f"No valid records loaded from: {src}")
    return records


def export_json(records: Iterable[STSRecord], out_path: str | Path) -> None:
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_csv(records: Iterable[STSRecord], out_path: str | Path) -> None:
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text1", "text2", "score"])
        writer.writeheader()
        writer.writerows(rows)


def export_records(
    records: Iterable[STSRecord],
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    if json_path is None and csv_path is None:
        raise ValueError("At least one output path must be provided")
    if json_path is not None:
        export_json(records, json_path)
    if csv_path is not None:
        export_csv(records, csv_path)
