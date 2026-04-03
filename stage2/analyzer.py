"""
Stage 2 Analyzer
================
汇总各方法 × 识别器的运行结果，生成：
  - dataset_methods_stat.json（每条样本的特征计数）
  - final_dataset.jsonl（去重后的最终四元组）
  - difference.md（Regular vs LLM 对比报告）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stage2.builder import ALL_METHODS, RunResult


def build_dataset_methods_stat(
    record_ids: List[str],
    record_pos: Dict[str, str],
    all_results: Dict[str, Dict[str, RunResult]],
) -> List[Dict]:
    """
    为每条样本生成各方法在两种识别路径下的特征计数汇总。
    all_results: {method_name: {"Regular": RunResult, "LLM": RunResult}}
    """
    rows = []
    for sid in record_ids:
        methods_feature_count: Dict[str, Dict] = {}
        total_regular = 0
        total_llm = 0
        for method in ALL_METHODS:
            reg_count = llm_count = 0
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
    将所有 Regular 路径下成功生成的样本合并为最终四元组列表。
    同一 sample_id 只保留第一个成功的方法（按 ALL_METHODS 顺序）。
    """
    seen: set = set()
    final: List[Dict] = []
    for method in ALL_METHODS:
        regular_result = all_results.get(method, {}).get("Regular")
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
    """生成 Regular vs LLM 对比 Markdown 报告。"""
    reg_stats = regular_result.stats
    llm_stats = llm_result.stats

    r_records = {rec["id"]: rec for rec in regular_result.records}
    l_records = {rec["id"]: rec for rec in llm_result.records}

    regular_better: list = []
    llm_better: list = []
    for sid in sorted(set(r_records) | set(l_records)):
        rr = r_records.get(sid)
        lr = l_records.get(sid)
        if rr and lr:
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
        f"| 成功率 | {reg_stats['success_ratio']*100:.1f}% | {llm_stats['success_ratio']*100:.1f}% |",
        f"| 平均特征数 | {reg_stats['avg_feature_count']:.2f} | {llm_stats['avg_feature_count']:.2f} |",
        f"| 处理时间 (s) | {reg_stats['processing_time_sec']} | {llm_stats['processing_time_sec']} |",
        "",
        "## 失败原因对比",
        "",
        "| 失败原因 | Regular | LLM |",
        "|---|---|---|",
    ]
    for reason in ("no_feature_found", "output_same_as_input", "empty_output", "exception"):
        rc = reg_stats["failure_reasons"].get(reason, 0)
        lc = llm_stats["failure_reasons"].get(reason, 0)
        lines.append(f"| {reason} | {rc} | {lc} |")

    lines += ["", "## 典型案例", ""]

    if regular_better:
        lines.append("### Regular 更优")
        for rr, lr in regular_better:
            lines += [
                f"- **输入**: {rr['pos'][:120]}",
                f"- **Regular**: 成功，{rr.get('replacement') or '—'}",
                f"- **LLM**: 失败，原因: {lr['failure_reason']}",
                "",
            ]

    if llm_better:
        lines.append("### LLM 更优")
        for rr, lr in llm_better:
            lines += [
                f"- **输入**: {rr['pos'][:120]}",
                f"- **Regular**: 失败，原因: {rr['failure_reason']}",
                f"- **LLM**: 成功，{lr.get('replacement') or '—'}",
                "",
            ]

    if not regular_better and not llm_better:
        lines += ["*（本次运行中未找到典型对比案例）*", ""]

    lines += [
        "## 结论",
        "",
        "- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子",
        "- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子",
    ]

    return "\n".join(lines)
