from __future__ import annotations

import pytest

from stage2.builder import RunResult, ALL_METHODS
from stage2.analyzer import (
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
            "failure_reasons": {
                "no_feature_found": failed, "output_same_as_input": 0,
                "empty_output": 0, "exception": 0,
            },
            "processing_time_sec": 1.5,
        },
        feature_counts={"s1": 2, "s2": 0},
    )


# ── build_dataset_methods_stat ─────────────────────────────────────────────

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
    rows = build_dataset_methods_stat(["s1"], {"s1": "text"}, {})
    for method in ALL_METHODS:
        assert rows[0]["methods_feature_count"][method]["Regular"] == 0
        assert rows[0]["methods_feature_count"][method]["LLM"] == 0


def test_build_dataset_methods_stat_llm_totals():
    reg = _make_run_result("numeric_metric_transform", "Regular", [], success_count=0)
    reg.feature_counts = {"s1": 2}
    llm = _make_run_result("numeric_metric_transform", "LLM", [], success_count=0)
    llm.feature_counts = {"s1": 3}
    all_results = {"numeric_metric_transform": {"Regular": reg, "LLM": llm}}
    rows = build_dataset_methods_stat(["s1"], {"s1": "text"}, all_results)
    assert rows[0]["total_features_llm"] == 3
    assert rows[0]["methods_feature_count"]["numeric_metric_transform"]["LLM"] == 3


# ── aggregate_final_dataset ────────────────────────────────────────────────

def test_aggregate_final_dataset_includes_only_successes():
    records = [
        {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
         "hard_neg": "h1", "method": "m", "recognizer": "Regular", "success": True,
         "replacement": "x→y", "failure_reason": None, "extraction_error": None},
        {"id": "s2", "anchor": "a", "pos": "p", "neg": "n",
         "hard_neg": None, "method": "m", "recognizer": "Regular", "success": False,
         "replacement": None, "failure_reason": "no_feature_found", "extraction_error": None},
    ]
    result = _make_run_result("numeric_metric_transform", "Regular", records, success_count=1)
    rows = aggregate_final_dataset({"numeric_metric_transform": {"Regular": result}})
    assert len(rows) == 1
    assert rows[0]["id"] == "s1"
    assert rows[0]["hard_neg"] == "h1"


def test_aggregate_final_dataset_deduplicates_by_sample_id():
    rec = {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
           "hard_neg": "h1", "method": "m1", "recognizer": "Regular", "success": True,
           "replacement": None, "failure_reason": None, "extraction_error": None}
    rec2 = {"id": "s1", "anchor": "a", "pos": "p", "neg": "n",
            "hard_neg": "h2", "method": "m2", "recognizer": "Regular", "success": True,
            "replacement": None, "failure_reason": None, "extraction_error": None}
    r1 = _make_run_result("numeric_metric_transform", "Regular", [rec], 1)
    r2 = _make_run_result("direct_negation_attack", "Regular", [rec2], 1)
    rows = aggregate_final_dataset({
        "numeric_metric_transform": {"Regular": r1},
        "direct_negation_attack": {"Regular": r2},
    })
    assert len(rows) == 1
    assert rows[0]["hard_neg"] == "h1"  # 第一个方法优先


# ── generate_difference_report ─────────────────────────────────────────────

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
    assert "成功率" in report


def test_generate_difference_report_shows_regular_better():
    r_rec = [{"id": "s1", "pos": "The price rose from 20 to 30.", "success": True,
               "replacement": "20 → 21", "failure_reason": None}]
    l_rec = [{"id": "s1", "pos": "The price rose from 20 to 30.", "success": False,
               "replacement": None, "failure_reason": "no_feature_found"}]
    reg = _make_run_result("numeric_metric_transform", "Regular", r_rec, 1)
    llm = _make_run_result("numeric_metric_transform", "LLM", l_rec, 0)
    report = generate_difference_report("numeric_metric_transform", reg, llm)
    assert "Regular 更优" in report


def test_generate_difference_report_shows_llm_better():
    r_rec = [{"id": "s1", "pos": "Michael went to Paris.", "success": False,
               "replacement": None, "failure_reason": "no_feature_found"}]
    l_rec = [{"id": "s1", "pos": "Michael went to Paris.", "success": True,
               "replacement": "Michael → John", "failure_reason": None}]
    reg = _make_run_result("entity_pronoun_substitution", "Regular", r_rec, 0)
    llm = _make_run_result("entity_pronoun_substitution", "LLM", l_rec, 1)
    report = generate_difference_report("entity_pronoun_substitution", reg, llm)
    assert "LLM 更优" in report


def test_generate_difference_report_replacement_none_shows_dash():
    r_rec = [{"id": "s1", "pos": "t", "success": True,
               "replacement": None, "failure_reason": None}]
    l_rec = [{"id": "s1", "pos": "t", "success": False,
               "replacement": None, "failure_reason": "no_feature_found"}]
    reg = _make_run_result("premise_disruption", "Regular", r_rec, 1)
    llm = _make_run_result("premise_disruption", "LLM", l_rec, 0)
    report = generate_difference_report("premise_disruption", reg, llm)
    assert "None" not in report   # replacement=None 不得直接打印为 "None"
    assert "—" in report          # 应显示破折号占位
