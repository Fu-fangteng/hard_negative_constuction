# Stage 2 Hard Negative Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin adapter layer (`src_v2/`) over existing `src/` to construct NLI four-tuple hard-negative training sets with per-method Regular/LLM dual-path tracking and structured logging.

**Architecture:** `src_v2/` imports directly from `src/` (constructors, llm_engine, formatter internals, data_utils) — zero modifications to Stage 1 code. A `PipelineRunner` processes one method × one recognizer per run, writing structured JSONL logs and summaries. An `Analyzer` aggregates all runs into global stats and difference reports.

**Tech Stack:** Python 3.9+, pandas + pyarrow (parquet), existing src/ modules, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add pyarrow |
| `src_v2/__init__.py` | Create | Package marker |
| `src_v2/data_loader.py` | Create | NLIRecord dataclass + parquet load/sample/save |
| `src_v2/feature_extractor.py` | Create | extract_regular, extract_llm, count_method_features |
| `src_v2/builder.py` | Create | PipelineRunner, RunResult, _diff_summary |
| `src_v2/analyzer.py` | Create | build_dataset_methods_stat, aggregate_final_dataset, generate_difference_report |
| `scripts_v2/__init__.py` | Create | Package marker |
| `scripts_v2/run_stage2.py` | Create | CLI entry point |
| `tests/__init__.py` | Create | Package marker |
| `tests/test_stage2/__init__.py` | Create | Package marker |
| `tests/test_stage2/test_data_loader.py` | Create | Tests for data_loader |
| `tests/test_stage2/test_feature_extractor.py` | Create | Tests for feature_extractor |
| `tests/test_stage2/test_builder.py` | Create | Tests for PipelineRunner |
| `tests/test_stage2/test_analyzer.py` | Create | Tests for analyzer functions |

---

## Task 1: Project Scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `src_v2/__init__.py`, `scripts_v2/__init__.py`
- Create: `tests/__init__.py`, `tests/test_stage2/__init__.py`

- [ ] **Step 1: Add pyarrow to requirements.txt**

Open `requirements.txt` and append:
```
pyarrow         # parquet backend for pandas
```

- [ ] **Step 2: Create package markers**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
mkdir -p src_v2 scripts_v2 tests/test_stage2
touch src_v2/__init__.py scripts_v2/__init__.py
touch tests/__init__.py tests/test_stage2/__init__.py
```

- [ ] **Step 3: Verify imports work**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -c "import src_v2; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src_v2/__init__.py scripts_v2/__init__.py tests/__init__.py tests/test_stage2/__init__.py
git commit -m "chore: scaffold src_v2, scripts_v2, tests directories for stage2"
```

---

## Task 2: data_loader.py

**Files:**
- Create: `src_v2/data_loader.py`
- Create: `tests/test_stage2/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage2/test_data_loader.py`:

```python
from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.data_loader import NLIRecord, load_parquet, save_preprocessed


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


def test_save_preprocessed_writes_json(sample_parquet, tmp_path):
    records = load_parquet(sample_parquet)
    out = tmp_path / "preprocessed_data.json"
    save_preprocessed(records, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert data[0] == {"id": "sample_000001", "anchor": "The cat sat on the mat.",
                       "pos": "A cat was sitting on a mat.", "neg": "The sky is blue."}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_data_loader.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'NLIRecord'`

- [ ] **Step 3: Implement src_v2/data_loader.py**

Create `src_v2/data_loader.py`:

```python
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import normalize_text


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
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    records: List[NLIRecord] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        records.append(NLIRecord(
            id=f"sample_{idx:06d}",
            anchor=normalize_text(getattr(row, "anchor", "")),
            pos=normalize_text(getattr(row, "positive", "")),
            neg=normalize_text(getattr(row, "negative", "")),
        ))
    return records


def save_preprocessed(records: List[NLIRecord], output_path: str) -> None:
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_data_loader.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src_v2/data_loader.py tests/test_stage2/test_data_loader.py
git commit -m "feat(stage2): add NLIRecord dataclass and parquet data loader"
```

---

## Task 3: feature_extractor.py

**Files:**
- Create: `src_v2/feature_extractor.py`
- Create: `tests/test_stage2/test_feature_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage2/test_feature_extractor.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.feature_extractor import extract_regular, extract_llm, count_method_features


def test_extract_regular_finds_numbers():
    feats = extract_regular("The price rose from $20 to $30 in 5 days.")
    assert len(feats.get("numbers", [])) >= 2


def test_extract_regular_finds_negations():
    feats = extract_regular("He did not go to the park.")
    assert "not" in feats.get("negations", [])


def test_extract_regular_finds_degree_words():
    feats = extract_regular("All students must complete the exam.")
    assert "all" in feats.get("degree_words", [])
    assert "must" in feats.get("degree_words", [])


def test_extract_regular_finds_logic_words():
    feats = extract_regular("She passed because she studied hard.")
    assert "because" in feats.get("logic_words", [])


def test_extract_llm_returns_empty_when_no_engine():
    feats = extract_llm("Any text.", llm_engine=None)
    assert feats == {}


def test_count_numeric_metric_transform():
    feats = {"numbers": ["$20", "$30", "5"]}
    assert count_method_features(feats, "numeric_metric_transform") == 3


def test_count_entity_pronoun_substitution():
    feats = {"entities": ["Michael", "Paris"], "pronouns": ["he"]}
    assert count_method_features(feats, "entity_pronoun_substitution") == 3


def test_count_direct_negation_attack_no_negation():
    feats = {"negations": []}
    assert count_method_features(feats, "direct_negation_attack") == 1


def test_count_direct_negation_attack_has_negation():
    feats = {"negations": ["not"]}
    assert count_method_features(feats, "direct_negation_attack") == 0


def test_count_premise_disruption_always_one():
    assert count_method_features({}, "premise_disruption") == 1


def test_count_role_swap():
    feats = {"subject_candidates": ["dog"], "object_candidates": ["man", "ball"]}
    assert count_method_features(feats, "role_swap") == 1


def test_count_unknown_method_returns_zero():
    assert count_method_features({}, "nonexistent_method") == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_feature_extractor.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'extract_regular'`

- [ ] **Step 3: Implement src_v2/feature_extractor.py**

Create `src_v2/feature_extractor.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.formatter import (
    _llm_extract,
    _merge_features,
    _regex_extract,
    _safe_spacy_extract,
)
from src.llm_engine import LocalLLMEngine


def extract_regular(text: str) -> Dict[str, List[str]]:
    """Merge regex + spaCy features (no LLM). Used for the Regular recognizer path."""
    return _merge_features(_regex_extract(text), _safe_spacy_extract(text))


def extract_llm(text: str, llm_engine: Optional[LocalLLMEngine]) -> Dict[str, List[str]]:
    """LLM-based feature extraction. Returns {} when llm_engine is None."""
    if llm_engine is None:
        return {}
    return _llm_extract(text, llm_engine)


def count_method_features(features: Dict[str, Any], method_name: str) -> int:
    """
    Return the number of usable features for the given method.
    Used to populate dataset_methods_stat.json.
    """
    f = features  # shorthand
    mapping = {
        "numeric_metric_transform":    len(f.get("numbers", [])),
        "entity_pronoun_substitution": len(f.get("entities", [])) + len(f.get("pronouns", [])),
        "scope_degree_scaling":        len(f.get("degree_words", [])),
        "direct_negation_attack":      0 if f.get("negations") else 1,
        "double_negation_attack":      len(f.get("negations", [])),
        "logical_operator_rewrite":    len(f.get("logic_words", [])),
        "role_swap":                   min(
                                           len(f.get("subject_candidates", [])),
                                           len(f.get("object_candidates", [])),
                                       ),
        "temporal_causal_inversion":   len(f.get("sequence_words", [])),
        "concept_hierarchy_shift":     len(f.get("entities", [])),
        "premise_disruption":          1,
    }
    return mapping.get(method_name, 0)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_feature_extractor.py -v
```

Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src_v2/feature_extractor.py tests/test_stage2/test_feature_extractor.py
git commit -m "feat(stage2): add feature extractor with Regular/LLM split and method feature counter"
```

---

## Task 4: builder.py

**Files:**
- Create: `src_v2/builder.py`
- Create: `tests/test_stage2/test_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage2/test_builder.py`:

```python
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.data_loader import NLIRecord
from src_v2.builder import PipelineRunner, RunResult


@pytest.fixture
def nli_records():
    return [
        NLIRecord("sample_000001", "anchor1", "The price rose from $20 to $30.", "neg1"),
        NLIRecord("sample_000002", "anchor2", "She went to the park.", "neg2"),
        NLIRecord("sample_000003", "anchor3", "All students must pass the exam.", "neg3"),
    ]


def test_pipeline_runner_creates_output_files(nli_records, tmp_path):
    out_dir = tmp_path / "numeric_metric_transform" / "Regular"
    runner = PipelineRunner(nli_records, "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()

    assert (out_dir / "constructed_data.json").exists()
    assert (out_dir / "method_stat.json").exists()
    assert (out_dir / "construction_log.jsonl").exists()
    assert (out_dir / "construction_summary.txt").exists()


def test_pipeline_runner_returns_run_result(nli_records, tmp_path):
    out_dir = tmp_path / "direct_negation_attack" / "Regular"
    runner = PipelineRunner(nli_records, "direct_negation_attack", "Regular", out_dir)
    result = runner.run()

    assert isinstance(result, RunResult)
    assert result.method_name == "direct_negation_attack"
    assert result.recognizer_type == "Regular"
    assert len(result.records) == 3
    assert isinstance(result.stats, dict)
    assert isinstance(result.feature_counts, dict)


def test_pipeline_runner_stats_counts_add_up(nli_records, tmp_path):
    out_dir = tmp_path / "scope_degree_scaling" / "Regular"
    runner = PipelineRunner(nli_records, "scope_degree_scaling", "Regular", out_dir)
    result = runner.run()

    s = result.stats
    total_failures = sum(s["failure_reasons"].values())
    assert s["success_count"] + total_failures == s["total_samples"]
    assert s["total_samples"] == 3


def test_pipeline_runner_numeric_succeeds_on_number_sentence(tmp_path):
    records = [NLIRecord("s1", "a", "The temperature rose from 20C to 30C.", "n")]
    out_dir = tmp_path / "numeric" / "Regular"
    runner = PipelineRunner(records, "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()

    assert result.stats["success_count"] == 1
    assert result.records[0]["success"] is True
    assert result.records[0]["hard_neg"] is not None
    assert result.records[0]["hard_neg"] != "The temperature rose from 20C to 30C."


def test_pipeline_runner_failure_reason_recorded(tmp_path):
    # A sentence with no numbers should fail numeric_metric_transform
    records = [NLIRecord("s1", "a", "The cat sat on the mat.", "n")]
    out_dir = tmp_path / "numeric_fail" / "Regular"
    runner = PipelineRunner(records, "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()

    assert result.records[0]["success"] is False
    assert result.records[0]["failure_reason"] == "no_feature_found"
    assert result.records[0]["hard_neg"] is None


def test_construction_log_is_valid_jsonl(nli_records, tmp_path):
    out_dir = tmp_path / "log_test" / "Regular"
    runner = PipelineRunner(nli_records, "direct_negation_attack", "Regular", out_dir)
    runner.run()

    log_path = out_dir / "construction_log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        entry = json.loads(line)
        assert "ts" in entry
        assert "sample_id" in entry
        assert "success" in entry
        assert "failure_reason" in entry
        assert "time_ms" in entry


def test_feature_counts_populated_for_all_records(nli_records, tmp_path):
    out_dir = tmp_path / "feat_count" / "Regular"
    runner = PipelineRunner(nli_records, "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()

    assert set(result.feature_counts.keys()) == {"sample_000001", "sample_000002", "sample_000003"}
    # sample_000001 has numbers, should have feature_count > 0
    assert result.feature_counts["sample_000001"] > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_builder.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'PipelineRunner'`

- [ ] **Step 3: Implement src_v2/builder.py**

Create `src_v2/builder.py`:

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constructors import apply_method
from src.llm_engine import LocalLLMEngine
from src.main_generator import ensure_text3_valid
from src_v2.data_loader import NLIRecord
from src_v2.feature_extractor import count_method_features, extract_llm, extract_regular

ALL_METHODS: List[str] = [
    "numeric_metric_transform",
    "entity_pronoun_substitution",
    "scope_degree_scaling",
    "direct_negation_attack",
    "double_negation_attack",
    "logical_operator_rewrite",
    "role_swap",
    "temporal_causal_inversion",
    "concept_hierarchy_shift",
    "premise_disruption",
]

_FAILURE_REASONS = ("no_feature_found", "output_same_as_input", "empty_output", "llm_error")


def _diff_summary(original: str, modified: str) -> Optional[str]:
    """Return a short 'token → token' string showing the first changed token."""
    orig_tokens = original.split()
    mod_tokens = modified.split()
    for o, m in zip(orig_tokens, mod_tokens):
        if o != m:
            return f"{o} → {m}"
    if len(orig_tokens) != len(mod_tokens):
        return f"[length {len(orig_tokens)} → {len(mod_tokens)} tokens]"
    return None


@dataclass
class RunResult:
    method_name: str
    recognizer_type: str
    records: List[Dict]
    stats: Dict
    feature_counts: Dict[str, int]  # {sample_id: feature_count for this method}


class PipelineRunner:
    def __init__(
        self,
        records: List[NLIRecord],
        method_name: str,
        recognizer_type: str,
        output_dir: Path,
        llm_engine: Optional[LocalLLMEngine] = None,
    ) -> None:
        self.records = records
        self.method_name = method_name
        self.recognizer_type = recognizer_type
        self.output_dir = Path(output_dir)
        self.llm_engine = llm_engine

    def run(self) -> RunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "construction_log.jsonl"

        constructed_records: List[Dict] = []
        feature_counts: Dict[str, int] = {}
        failure_reasons: Dict[str, int] = {r: 0 for r in _FAILURE_REASONS}
        total_feat_count = 0
        start_time = time.time()

        with log_path.open("w", encoding="utf-8") as log_f:
            for record in self.records:
                t0 = time.time()

                # --- Feature extraction ---
                try:
                    if self.recognizer_type == "Regular":
                        features = extract_regular(record.pos)
                    else:
                        features = extract_llm(record.pos, self.llm_engine)
                except Exception:
                    features = {}

                feat_count = count_method_features(features, self.method_name)
                feature_counts[record.id] = feat_count
                total_feat_count += feat_count

                # --- Construction ---
                hard_neg: Optional[str] = None
                replacement: Optional[str] = None
                failure_reason: Optional[str] = None

                try:
                    raw_out = apply_method(
                        method_name=self.method_name,
                        formatted_item={},
                        text=record.pos,
                        features=features,
                        llm_engine=self.llm_engine if self.recognizer_type == "LLM" else None,
                    )
                    if raw_out is None:
                        failure_reason = "no_feature_found"
                        failure_reasons["no_feature_found"] += 1
                    else:
                        validated = ensure_text3_valid(record.pos, raw_out)
                        if validated is None:
                            if not raw_out.strip():
                                failure_reason = "empty_output"
                                failure_reasons["empty_output"] += 1
                            else:
                                failure_reason = "output_same_as_input"
                                failure_reasons["output_same_as_input"] += 1
                        else:
                            hard_neg = validated
                            replacement = _diff_summary(record.pos, validated)
                except Exception:
                    failure_reason = "llm_error"
                    failure_reasons["llm_error"] += 1

                elapsed_ms = int((time.time() - t0) * 1000)

                constructed_records.append({
                    "id": record.id,
                    "anchor": record.anchor,
                    "pos": record.pos,
                    "neg": record.neg,
                    "hard_neg": hard_neg,
                    "method": self.method_name,
                    "recognizer": self.recognizer_type,
                    "success": hard_neg is not None,
                    "replacement": replacement,
                    "failure_reason": failure_reason,
                })

                log_entry = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "sample_id": record.id,
                    "method": self.method_name,
                    "recognizer": self.recognizer_type,
                    "input": record.pos,
                    "features_found": {k: v for k, v in features.items() if v},
                    "replacement": replacement,
                    "output": hard_neg,
                    "success": hard_neg is not None,
                    "failure_reason": failure_reason,
                    "time_ms": elapsed_ms,
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        elapsed_total = round(time.time() - start_time, 2)
        total = len(constructed_records)
        success_count = sum(1 for r in constructed_records if r["success"])

        stats = {
            "method_name": self.method_name,
            "recognizer_type": self.recognizer_type,
            "total_samples": total,
            "success_count": success_count,
            "success_ratio": round(success_count / total, 4) if total else 0.0,
            "avg_feature_count": round(total_feat_count / total, 2) if total else 0.0,
            "failure_reasons": failure_reasons,
            "processing_time_sec": elapsed_total,
        }

        with (self.output_dir / "constructed_data.json").open("w", encoding="utf-8") as f:
            json.dump(constructed_records, f, ensure_ascii=False, indent=2)

        with (self.output_dir / "method_stat.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self._write_summary(stats, failure_reasons)

        return RunResult(
            method_name=self.method_name,
            recognizer_type=self.recognizer_type,
            records=constructed_records,
            stats=stats,
            feature_counts=feature_counts,
        )

    def _write_summary(self, stats: Dict, failure_reasons: Dict) -> None:
        total = stats["total_samples"]
        success = stats["success_count"]
        failed = total - success

        def pct(n: int, d: int) -> str:
            return f"({n / d * 100:.1f}%)" if d else "(0.0%)"

        lines = [
            "==============================",
            f"Method    : {stats['method_name']}",
            f"Recognizer: {stats['recognizer_type']}",
            f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "------------------------------",
            f"Total     : {total}",
            f"Success   : {success}  {pct(success, total)}",
            f"Failed    : {failed}  {pct(failed, total)}",
            "",
            "Failure breakdown:",
        ]
        for reason, count in failure_reasons.items():
            lines.append(f"  {reason:<25}: {count:4d}  {pct(count, failed)}")
        lines += [
            "",
            f"Avg feature count     : {stats['avg_feature_count']:.2f}",
            f"Processing time       : {stats['processing_time_sec']}s",
            "==============================",
        ]
        (self.output_dir / "construction_summary.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_builder.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src_v2/builder.py tests/test_stage2/test_builder.py
git commit -m "feat(stage2): add PipelineRunner with structured JSONL logging and RunResult"
```

---

## Task 5: analyzer.py

**Files:**
- Create: `src_v2/analyzer.py`
- Create: `tests/test_stage2/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage2/test_analyzer.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.builder import RunResult, ALL_METHODS
from src_v2.analyzer import (
    build_dataset_methods_stat,
    aggregate_final_dataset,
    generate_difference_report,
)


def _make_run_result(method, recognizer, records, success_count=1):
    total = len(records)
    failed = total - success_count
    return RunResult(
        method_name=method,
        recognizer_type=recognizer,
        records=records,
        stats={
            "method_name": method,
            "recognizer_type": recognizer,
            "total_samples": total,
            "success_count": success_count,
            "success_ratio": success_count / total if total else 0,
            "avg_feature_count": 1.2,
            "failure_reasons": {"no_feature_found": failed, "output_same_as_input": 0,
                                 "empty_output": 0, "llm_error": 0},
            "processing_time_sec": 1.5,
        },
        feature_counts={"s1": 2, "s2": 0},
    )


# --- build_dataset_methods_stat ---

def test_build_dataset_methods_stat_covers_all_samples():
    reg = _make_run_result("numeric_metric_transform", "Regular", [], success_count=0)
    reg.feature_counts = {"s1": 3, "s2": 0}
    all_results = {"numeric_metric_transform": {"Regular": reg}}
    rows = build_dataset_methods_stat(["s1", "s2"], {"s1": "text1", "s2": "text2"}, all_results)
    assert len(rows) == 2
    assert rows[0]["id"] == "s1"
    assert rows[0]["methods_feature_count"]["numeric_metric_transform"]["Regular"] == 3
    assert rows[1]["methods_feature_count"]["numeric_metric_transform"]["Regular"] == 0


def test_build_dataset_methods_stat_sums_totals():
    reg = _make_run_result("numeric_metric_transform", "Regular", [], success_count=0)
    reg.feature_counts = {"s1": 2}
    neg_reg = _make_run_result("direct_negation_attack", "Regular", [], success_count=0)
    neg_reg.feature_counts = {"s1": 1}
    all_results = {
        "numeric_metric_transform": {"Regular": reg},
        "direct_negation_attack": {"Regular": neg_reg},
    }
    rows = build_dataset_methods_stat(["s1"], {"s1": "text"}, all_results)
    assert rows[0]["total_features_regular"] == 3


def test_build_dataset_methods_stat_missing_method_counts_zero():
    all_results = {}
    rows = build_dataset_methods_stat(["s1"], {"s1": "text"}, all_results)
    for method in ALL_METHODS:
        assert rows[0]["methods_feature_count"][method]["Regular"] == 0
        assert rows[0]["methods_feature_count"][method]["LLM"] == 0


# --- aggregate_final_dataset ---

def test_aggregate_final_dataset_includes_only_successes():
    records = [
        {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
         "hard_neg": "h1", "method": "m", "recognizer": "Regular", "success": True,
         "replacement": "x→y", "failure_reason": None},
        {"id": "s2", "anchor": "a", "pos": "p", "neg": "n",
         "hard_neg": None, "method": "m", "recognizer": "Regular", "success": False,
         "replacement": None, "failure_reason": "no_feature_found"},
    ]
    result = _make_run_result("numeric_metric_transform", "Regular", records, success_count=1)
    rows = aggregate_final_dataset({"numeric_metric_transform": {"Regular": result}})
    assert len(rows) == 1
    assert rows[0]["id"] == "s1"
    assert rows[0]["hard_neg"] == "h1"


def test_aggregate_final_dataset_deduplicates_by_sample_id():
    rec = {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
           "hard_neg": "h1", "method": "m1", "recognizer": "Regular", "success": True,
           "replacement": None, "failure_reason": None}
    rec2 = {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
            "hard_neg": "h2", "method": "m2", "recognizer": "Regular", "success": True,
            "replacement": None, "failure_reason": None}
    r1 = _make_run_result("numeric_metric_transform", "Regular", [rec], 1)
    r2 = _make_run_result("direct_negation_attack", "Regular", [rec2], 1)
    rows = aggregate_final_dataset({
        "numeric_metric_transform": {"Regular": r1},
        "direct_negation_attack": {"Regular": r2},
    })
    # s1 should appear only once (first method wins)
    assert len(rows) == 1
    assert rows[0]["hard_neg"] == "h1"


# --- generate_difference_report ---

def test_generate_difference_report_contains_method_name():
    r_rec = [{"id": "s1", "pos": "text", "success": True,
               "replacement": "a→b", "failure_reason": None}]
    l_rec = [{"id": "s1", "pos": "text", "success": False,
               "replacement": None, "failure_reason": "no_feature_found"}]
    reg = _make_run_result("numeric_metric_transform", "Regular", r_rec, 1)
    llm = _make_run_result("numeric_metric_transform", "LLM", l_rec, 0)
    report = generate_difference_report("numeric_metric_transform", reg, llm)
    assert "numeric_metric_transform" in report
    assert "Regular" in report
    assert "LLM" in report


def test_generate_difference_report_contains_stats_table():
    r_rec = [{"id": "s1", "pos": "t", "success": True, "replacement": "x→y", "failure_reason": None}]
    l_rec = [{"id": "s1", "pos": "t", "success": False, "replacement": None, "failure_reason": "no_feature_found"}]
    reg = _make_run_result("direct_negation_attack", "Regular", r_rec, 1)
    llm = _make_run_result("direct_negation_attack", "LLM", l_rec, 0)
    report = generate_difference_report("direct_negation_attack", reg, llm)
    assert "成功率" in report or "success_ratio" in report.lower() or "100.0%" in report
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_analyzer.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'build_dataset_methods_stat'`

- [ ] **Step 3: Implement src_v2/analyzer.py**

Create `src_v2/analyzer.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.builder import ALL_METHODS, RunResult


def build_dataset_methods_stat(
    record_ids: List[str],
    record_pos: Dict[str, str],
    all_results: Dict[str, Dict[str, RunResult]],
) -> List[Dict]:
    """
    For each sample, record feature counts per method per recognizer.
    all_results: {method_name: {"Regular": RunResult, "LLM": RunResult}}
    """
    rows = []
    for sid in record_ids:
        methods_feature_count: Dict[str, Dict] = {}
        total_regular = 0
        total_llm = 0
        for method in ALL_METHODS:
            reg_count = 0
            llm_count = 0
            method_results = all_results.get(method, {})
            if "Regular" in method_results:
                reg_count = method_results["Regular"].feature_counts.get(sid, 0)
            if "LLM" in method_results:
                llm_count = method_results["LLM"].feature_counts.get(sid, 0)
            methods_feature_count[method] = {"Regular": reg_count, "LLM": llm_count}
            total_regular += reg_count
            total_llm += llm_count
        rows.append({
            "id": sid,
            "pos": record_pos.get(sid, ""),
            "methods_feature_count": methods_feature_count,
            "total_features_regular": total_regular,
            "total_features_llm": total_llm,
        })
    return rows


def aggregate_final_dataset(
    all_results: Dict[str, Dict[str, RunResult]],
) -> List[Dict]:
    """
    Merge all Regular-path successful samples into one deduplicated list.
    First method to produce a hard_neg for a sample_id wins.
    """
    seen: set = set()
    final: List[Dict] = []
    for method in ALL_METHODS:
        method_results = all_results.get(method, {})
        regular_result = method_results.get("Regular")
        if regular_result is None:
            continue
        for rec in regular_result.records:
            if not rec["success"]:
                continue
            sid = rec["id"]
            if sid in seen:
                continue
            seen.add(sid)
            final.append({
                "id": sid,
                "anchor": rec["anchor"],
                "pos": rec["pos"],
                "neg": rec["neg"],
                "hard_neg": rec["hard_neg"],
                "method": rec["method"],
                "recognizer": rec["recognizer"],
            })
    return final


def generate_difference_report(
    method_name: str,
    regular_result: RunResult,
    llm_result: RunResult,
) -> str:
    r = regular_result.stats
    l = llm_result.stats

    r_records = {rec["id"]: rec for rec in regular_result.records}
    l_records = {rec["id"]: rec for rec in llm_result.records}

    regular_better: list = []
    llm_better: list = []
    for sid in r_records:
        if sid not in l_records:
            continue
        rr, lr = r_records[sid], l_records[sid]
        if rr["success"] and not lr["success"] and len(regular_better) < 2:
            regular_better.append((rr, lr))
        elif not rr["success"] and lr["success"] and len(llm_better) < 2:
            llm_better.append((rr, lr))

    lines = [
        f"# {method_name} — Regular vs LLM 对比报告",
        "",
        "## 总体统计",
        "",
        "| 指标 | Regular | LLM |",
        "|---|---|---|",
        f"| 成功率 | {r['success_ratio']*100:.1f}% | {l['success_ratio']*100:.1f}% |",
        f"| 平均特征数 | {r['avg_feature_count']:.2f} | {l['avg_feature_count']:.2f} |",
        f"| 处理时间 (s) | {r['processing_time_sec']} | {l['processing_time_sec']} |",
        "",
        "## 失败原因对比",
        "",
        "| 失败原因 | Regular | LLM |",
        "|---|---|---|",
    ]
    for reason in ("no_feature_found", "output_same_as_input", "empty_output", "llm_error"):
        rc = r["failure_reasons"].get(reason, 0)
        lc = l["failure_reasons"].get(reason, 0)
        lines.append(f"| {reason} | {rc} | {lc} |")

    lines += ["", "## 典型案例", ""]

    if regular_better:
        lines.append("### Regular 更优")
        for rr, lr in regular_better:
            lines += [
                f"- **输入**: {rr['pos'][:100]}",
                f"- **Regular**: 成功，{rr['replacement']}",
                f"- **LLM**: 失败，reason: {lr['failure_reason']}",
                "",
            ]

    if llm_better:
        lines.append("### LLM 更优")
        for rr, lr in llm_better:
            lines += [
                f"- **输入**: {rr['pos'][:100]}",
                f"- **Regular**: 失败，reason: {rr['failure_reason']}",
                f"- **LLM**: 成功，{lr['replacement']}",
                "",
            ]

    if not regular_better and not llm_better:
        lines += ["*（本次运行中未找到典型对比案例）*", ""]

    lines += [
        "## 结论",
        "",
        "- Regular 推荐场景：含明确数值/逻辑词的句子",
        "- LLM 推荐场景：含命名实体、复杂代词指代的句子",
    ]

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/test_analyzer.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src_v2/analyzer.py tests/test_stage2/test_analyzer.py
git commit -m "feat(stage2): add analyzer for dataset_methods_stat, final_dataset aggregation, difference reports"
```

---

## Task 6: run_stage2.py Script

**Files:**
- Create: `scripts_v2/run_stage2.py`

- [ ] **Step 1: Implement scripts_v2/run_stage2.py**

Create `scripts_v2/run_stage2.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_engine import GenerationConfig, LocalLLMEngine
from src_v2.analyzer import (
    aggregate_final_dataset,
    build_dataset_methods_stat,
    generate_difference_report,
)
from src_v2.builder import ALL_METHODS, PipelineRunner
from src_v2.data_loader import load_parquet, save_preprocessed


def _parse_methods(s: str) -> list:
    if s.strip().lower() == "all":
        return ALL_METHODS
    return [m.strip() for m in s.split(",") if m.strip()]


def _parse_recognizers(s: str) -> list:
    mapping = {"both": ["Regular", "LLM"], "regular": ["Regular"], "llm": ["LLM"]}
    return mapping.get(s.lower(), ["Regular"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: NLI Hard Negative Pipeline")
    parser.add_argument("--input_path",   required=True,          help="Path to .parquet file")
    parser.add_argument("--output_base",  default="Data/",        help="Root output directory")
    parser.add_argument("--sample_size",  type=int, default=1000, help="Samples to process (None=all)")
    parser.add_argument("--methods",      default="all",          help="Comma-separated methods or 'all'")
    parser.add_argument("--recognizer",   default="regular",
                        choices=["regular", "llm", "both"],       help="Feature recognizer(s) to use")
    parser.add_argument("--llm_model",    default="",             help="Local LLM model path (for LLM recognizer)")
    parser.add_argument("--seed",         type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    methods = _parse_methods(args.methods)
    recognizers = _parse_recognizers(args.recognizer)

    # Load LLM if needed
    llm_engine = None
    if "LLM" in recognizers:
        if not args.llm_model:
            print("[WARN] --recognizer includes LLM but --llm_model not provided. LLM path will be skipped.")
            recognizers = [r for r in recognizers if r != "LLM"]
        else:
            print(f"[LLM] Loading {args.llm_model}...")
            llm_engine = LocalLLMEngine(model_name_or_path=args.llm_model)
            llm_engine.load()
            print("[LLM] Ready.")

    # Step 1: Load & preprocess
    print(f"\n[STEP 1] Loading data from {args.input_path}...")
    records = load_parquet(args.input_path, sample_size=args.sample_size, seed=args.seed)
    print(f"  → {len(records)} samples loaded")

    preprocessed_path = output_base / "preprocessed_data" / "preprocessed_data.json"
    save_preprocessed(records, str(preprocessed_path))
    print(f"  → Saved to {preprocessed_path}")

    # Step 2: Run pipeline for each method × recognizer
    all_results: dict = {}

    for method in methods:
        all_results[method] = {}
        for recognizer in recognizers:
            out_dir = output_base / "processed_data" / method / recognizer
            print(f"\n[RUN] {method} × {recognizer}")
            runner = PipelineRunner(
                records=records,
                method_name=method,
                recognizer_type=recognizer,
                output_dir=out_dir,
                llm_engine=llm_engine if recognizer == "LLM" else None,
            )
            result = runner.run()
            all_results[method][recognizer] = result
            s = result.stats
            print(f"  → {s['success_count']}/{s['total_samples']} success ({s['success_ratio']*100:.1f}%)"
                  f"  time={s['processing_time_sec']}s")

        # Generate difference.md when both paths exist
        if "Regular" in all_results[method] and "LLM" in all_results[method]:
            diff = generate_difference_report(
                method,
                all_results[method]["Regular"],
                all_results[method]["LLM"],
            )
            diff_path = output_base / "processed_data" / method / "difference.md"
            diff_path.write_text(diff, encoding="utf-8")
            print(f"  → difference.md written to {diff_path}")

    # Step 3: Global aggregation
    print("\n[STEP 3] Aggregating global stats...")
    record_ids = [r.id for r in records]
    record_pos = {r.id: r.pos for r in records}

    stat_rows = build_dataset_methods_stat(record_ids, record_pos, all_results)
    stat_path = output_base / "processed_data" / "dataset_methods_stat.json"
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    with stat_path.open("w", encoding="utf-8") as f:
        json.dump(stat_rows, f, ensure_ascii=False, indent=2)
    print(f"  → dataset_methods_stat.json: {len(stat_rows)} samples")

    final_rows = aggregate_final_dataset(all_results)
    final_path = output_base / "processed_data" / "final_dataset.jsonl"
    with final_path.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  → final_dataset.jsonl: {len(final_rows)} unique samples")

    print(f"\n✅ Stage 2 complete! Output: {output_base / 'processed_data'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with dry run**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 scripts_v2/run_stage2.py \
  --input_path data/train-00000-of-00001.parquet \
  --output_base Data/ \
  --sample_size 1000 \
  --methods "numeric_metric_transform,direct_negation_attack" \
  --recognizer regular \
  --seed 42
```

Expected output:
```
[STEP 1] Loading data from data/train-00000-of-00001.parquet...
  → 1000 samples loaded
  → Saved to Data/preprocessed_data/preprocessed_data.json

[RUN] numeric_metric_transform × Regular
  → XX/1000 success (XX.X%)  time=X.Xs

[RUN] direct_negation_attack × Regular
  → XX/1000 success (XX.X%)  time=X.Xs

[STEP 3] Aggregating global stats...
  → dataset_methods_stat.json: 1000 samples
  → final_dataset.jsonl: XX unique samples

✅ Stage 2 complete! Output: Data/processed_data
```

- [ ] **Step 3: Verify output file structure**

```bash
find Data/ -type f | sort
```

Expected:
```
Data/preprocessed_data/preprocessed_data.json
Data/processed_data/dataset_methods_stat.json
Data/processed_data/direct_negation_attack/Regular/constructed_data.json
Data/processed_data/direct_negation_attack/Regular/construction_log.jsonl
Data/processed_data/direct_negation_attack/Regular/construction_summary.txt
Data/processed_data/direct_negation_attack/Regular/method_stat.json
Data/processed_data/final_dataset.jsonl
Data/processed_data/numeric_metric_transform/Regular/constructed_data.json
Data/processed_data/numeric_metric_transform/Regular/construction_log.jsonl
Data/processed_data/numeric_metric_transform/Regular/construction_summary.txt
Data/processed_data/numeric_metric_transform/Regular/method_stat.json
```

- [ ] **Step 4: Spot-check a summary file**

```bash
cat Data/processed_data/numeric_metric_transform/Regular/construction_summary.txt
```

Verify: success + all failure_reasons counts sum to 1000.

- [ ] **Step 5: Commit**

```bash
git add scripts_v2/run_stage2.py
git commit -m "feat(stage2): add run_stage2.py CLI entry point"
```

---

## Task 7: Full Test Suite Pass + Final Validation

**Files:** No new files. Run all tests.

- [ ] **Step 1: Run complete test suite**

```bash
cd /Users/feng/Downloads/MRL/Coding/hard_neg
python3 -m pytest tests/test_stage2/ -v
```

Expected: All tests pass (no failures).

- [ ] **Step 2: Validate method_stat integrity for all methods**

Run all 10 methods and verify stats add up:

```bash
python3 scripts_v2/run_stage2.py \
  --input_path data/train-00000-of-00001.parquet \
  --output_base Data/ \
  --sample_size 1000 \
  --methods all \
  --recognizer regular \
  --seed 42
```

Then check every method_stat.json:

```bash
python3 -c "
import json, pathlib
for p in sorted(pathlib.Path('Data/processed_data').glob('*/Regular/method_stat.json')):
    s = json.loads(p.read_text())
    total = s['total_samples']
    failures = sum(s['failure_reasons'].values())
    ok = s['success_count'] + failures == total
    print(f\"{'OK' if ok else 'FAIL'} {s['method_name']:35s} success={s['success_count']}/{total}\")
"
```

Expected: All lines start with `OK`.

- [ ] **Step 3: Validate dataset_methods_stat sample coverage**

```bash
python3 -c "
import json
rows = json.loads(open('Data/processed_data/dataset_methods_stat.json').read())
print(f'Samples in stat: {len(rows)}')
assert len(rows) == 1000, 'Coverage mismatch'
print('Coverage: OK')
"
```

Expected:
```
Samples in stat: 1000
Coverage: OK
```

- [ ] **Step 4: Commit**

```bash
git add Data/
git commit -m "feat(stage2): complete stage2 pipeline — 1000-sample regular run validated"
```

---

## Self-Review Checklist (completed inline)

**Spec coverage:**
- ✅ NLIRecord + parquet load → Task 2
- ✅ Regular/LLM feature split → Task 3
- ✅ PipelineRunner with construction_log.jsonl + construction_summary.txt → Task 4
- ✅ constructed_data.json + method_stat.json → Task 4
- ✅ failure_reason enum (4 values) → Task 4
- ✅ build_dataset_methods_stat with feature counts (not booleans) → Task 5
- ✅ aggregate_final_dataset with dedup → Task 5
- ✅ generate_difference_report → Task 5
- ✅ final_dataset.jsonl → Task 6
- ✅ CLI with --recognizer both/regular/llm → Task 6
- ✅ difference.md generated when both recognizers run → Task 6

**Placeholder scan:** None found.

**Type consistency:** `RunResult.feature_counts: Dict[str, int]` defined in Task 4, used correctly in Task 5. `ALL_METHODS` defined in `builder.py`, imported in `analyzer.py` and `run_stage2.py`. `NLIRecord` defined in `data_loader.py`, used in `builder.py` and `run_stage2.py`.
