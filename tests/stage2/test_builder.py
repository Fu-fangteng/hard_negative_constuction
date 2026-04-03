from __future__ import annotations

import json
import pytest

from stage2.data_loader import NLIRecord
from stage2.builder import PipelineRunner, RunResult, _diff_summary


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
    runner.run()
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
    records = [NLIRecord("s1", "a", "The temperature rose from 20 to 30 degrees.", "n")]
    out_dir = tmp_path / "numeric" / "Regular"
    runner = PipelineRunner(records, "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()
    assert result.stats["success_count"] == 1
    assert result.records[0]["success"] is True
    assert result.records[0]["hard_neg"] is not None
    assert result.records[0]["hard_neg"] != "The temperature rose from 20 to 30 degrees."


def test_pipeline_runner_entity_succeeds_on_name_sentence(tmp_path):
    """核心修复验证：含实体句子，entity_pronoun_substitution 必须成功。"""
    records = [NLIRecord("s1", "a", "Michael traveled to Paris last summer.", "n")]
    out_dir = tmp_path / "entity" / "Regular"
    runner = PipelineRunner(records, "entity_pronoun_substitution", "Regular", out_dir)
    result = runner.run()
    assert result.stats["success_count"] == 1, (
        f"Entity substitution failed. hard_neg={result.records[0]['hard_neg']!r}, "
        f"features={result.records[0].get('extraction_error')}"
    )
    assert result.records[0]["hard_neg"] != "Michael traveled to Paris last summer."


def test_pipeline_runner_failure_reason_recorded(tmp_path):
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
    assert result.feature_counts["sample_000001"] > 0


def test_pipeline_runner_empty_records(tmp_path):
    out_dir = tmp_path / "empty" / "Regular"
    runner = PipelineRunner([], "numeric_metric_transform", "Regular", out_dir)
    result = runner.run()
    assert result.stats["total_samples"] == 0
    assert result.stats["success_count"] == 0
    assert result.stats["success_ratio"] == 0.0
    assert result.stats["avg_feature_count"] == 0.0
    assert len(result.records) == 0


def test_diff_summary_finds_first_changed_token():
    result = _diff_summary("The price is 20 dollars.", "The price is 21 dollars.")
    assert result == "20 → 21"


def test_diff_summary_length_change():
    result = _diff_summary("He went.", "He did not go.")
    assert result is not None
    assert "tokens" in result or "→" in result
