"""
Stage 2 Builder (PipelineRunner)
=================================
对每条 NLI positive 句子应用指定构造方法，生成 hard_neg。
使用 Stage 2 自己的 feature_extractor 和 constructors
（不再直接依赖 Stage 1 内部模块）。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stage2.constructors import apply_method
from stage2.data_loader import NLIRecord
from stage2.feature_extractor import count_method_features, extract_llm, extract_regular

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

_FAILURE_REASONS = ("no_feature_found", "output_same_as_input", "empty_output", "exception")


def _validate(original: str, generated: Optional[str]) -> Optional[str]:
    if generated is None:
        return None
    g = " ".join(generated.strip().split())
    o = " ".join(original.strip().split())
    return generated.strip() if g and g != o else None


def _diff_summary(original: str, modified: str) -> Optional[str]:
    orig_toks = original.split()
    mod_toks = modified.split()
    for o, m in zip(orig_toks, mod_toks):
        if o != m:
            return f"{o} → {m}"
    if len(orig_toks) != len(mod_toks):
        return f"[length {len(orig_toks)} → {len(mod_toks)} tokens]"
    return None


@dataclass
class RunResult:
    method_name: str
    recognizer_type: str
    records: List[Dict]
    stats: Dict
    feature_counts: Dict[str, int]


class PipelineRunner:
    def __init__(
        self,
        records: List[NLIRecord],
        method_name: str,
        recognizer_type: str,        # "Regular" 或 "LLM"
        output_dir: Path,
        llm_engine: Optional[Any] = None,
    ) -> None:
        self.records = records
        self.method_name = method_name
        self.recognizer_type = recognizer_type
        self.output_dir = Path(output_dir)
        self.llm_engine = llm_engine

    def run(self) -> RunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "construction_log.jsonl"

        constructed: List[Dict] = []
        feature_counts: Dict[str, int] = {}
        failure_reasons: Dict[str, int] = {r: 0 for r in _FAILURE_REASONS}
        total_feat = 0
        t_start = time.time()

        with log_path.open("w", encoding="utf-8") as log_f:
            for record in self.records:
                t0 = time.time()

                # ── 特征提取 ──────────────────────────────────────────────
                extraction_error: Optional[str] = None
                try:
                    if self.recognizer_type == "Regular":
                        features = extract_regular(record.pos)
                    else:
                        features = extract_llm(record.pos, self.llm_engine)
                except Exception as exc:
                    features = {}
                    extraction_error = f"{type(exc).__name__}: {exc}"

                feat_count = count_method_features(features, self.method_name)
                feature_counts[record.id] = feat_count
                total_feat += feat_count

                # ── 构造 hard_neg ─────────────────────────────────────────
                hard_neg: Optional[str] = None
                replacement: Optional[str] = None
                failure_reason: Optional[str] = None

                try:
                    raw = apply_method(
                        method_name=self.method_name,
                        formatted_item={},
                        text=record.pos,
                        features=features,
                        llm_engine=self.llm_engine if self.recognizer_type == "LLM" else None,
                    )
                    if raw is None:
                        failure_reason = "no_feature_found"
                        failure_reasons["no_feature_found"] += 1
                    else:
                        validated = _validate(record.pos, raw)
                        if validated is None:
                            if not raw.strip():
                                failure_reason = "empty_output"
                                failure_reasons["empty_output"] += 1
                            else:
                                failure_reason = "output_same_as_input"
                                failure_reasons["output_same_as_input"] += 1
                        else:
                            hard_neg = validated
                            replacement = _diff_summary(record.pos, validated)
                except Exception as exc:
                    failure_reason = "exception"
                    failure_reasons["exception"] += 1
                    extraction_error = (extraction_error or "") + f" | construct: {type(exc).__name__}: {exc}"

                elapsed_ms = int((time.time() - t0) * 1000)

                row = {
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
                    "extraction_error": extraction_error,
                }
                constructed.append(row)

                log_f.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "sample_id": record.id,
                    "method": self.method_name,
                    "recognizer": self.recognizer_type,
                    "input": record.pos,
                    "features_found": {k: v for k, v in features.items() if v},
                    "replacement": replacement,
                    "output": hard_neg,
                    "success": hard_neg is not None,
                    "extraction_error": extraction_error,
                    "failure_reason": failure_reason,
                    "time_ms": elapsed_ms,
                }, ensure_ascii=False) + "\n")

        elapsed_total = round(time.time() - t_start, 2)
        total = len(constructed)
        success_count = sum(1 for r in constructed if r["success"])

        stats = {
            "method_name": self.method_name,
            "recognizer_type": self.recognizer_type,
            "total_samples": total,
            "success_count": success_count,
            "success_ratio": round(success_count / total, 4) if total else 0.0,
            "avg_feature_count": round(total_feat / total, 2) if total else 0.0,
            "failure_reasons": failure_reasons,
            "processing_time_sec": elapsed_total,
        }

        with (self.output_dir / "constructed_data.json").open("w", encoding="utf-8") as f:
            json.dump(constructed, f, ensure_ascii=False, indent=2)

        with (self.output_dir / "method_stat.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self._write_summary(stats, failure_reasons)

        return RunResult(
            method_name=self.method_name,
            recognizer_type=self.recognizer_type,
            records=constructed,
            stats=stats,
            feature_counts=feature_counts,
        )

    def _write_summary(self, stats: Dict, failure_reasons: Dict) -> None:
        total   = stats["total_samples"]
        success = stats["success_count"]
        failed  = total - success

        def pct(n: int, d: int) -> str:
            return f"({n / d * 100:.1f}%)" if d else "(0.0%)"

        lines = [
            "=" * 40,
            f"Method    : {stats['method_name']}",
            f"Recognizer: {stats['recognizer_type']}",
            f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 40,
            f"Total     : {total}",
            f"Success   : {success}  {pct(success, total)}",
            f"Failed    : {failed}  {pct(failed, total)}",
            "",
            "Failure breakdown:",
        ]
        if failed == 0:
            lines.append("  (none)")
        else:
            for reason, count in failure_reasons.items():
                lines.append(f"  {reason:<25}: {count:4d}  {pct(count, failed)}")
        lines += [
            "",
            f"Avg feature count : {stats['avg_feature_count']:.2f}",
            f"Processing time   : {stats['processing_time_sec']}s",
            "=" * 40,
        ]
        (self.output_dir / "construction_summary.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
