from __future__ import annotations

import pytest
from stage2.feature_extractor import (
    extract_regular,
    extract_llm,
    count_method_features,
    _extract_entities_regex,
)


# ── 正则实体识别（核心修复点）──────────────────────────────────────────────

def test_extract_entities_regex_person_name():
    entities = _extract_entities_regex("Michael went to the store yesterday.")
    assert "Michael" in entities


def test_extract_entities_regex_place_name():
    entities = _extract_entities_regex("She traveled from Paris to London.")
    assert "Paris" in entities
    assert "London" in entities


def test_extract_entities_regex_multi_word_name():
    entities = _extract_entities_regex("New York is a great city.")
    assert "New York" in entities


def test_extract_entities_regex_title_plus_name():
    entities = _extract_entities_regex("Dr. Smith examined the patient.")
    assert any("Smith" in e for e in entities)


def test_extract_entities_regex_acronym():
    entities = _extract_entities_regex("The NASA scientist worked at WHO.")
    assert "NASA" in entities
    assert "WHO" in entities


def test_extract_entities_regex_skips_sentence_start():
    # "The" at sentence start should NOT be captured as entity
    entities = _extract_entities_regex("The dog is running fast.")
    assert "The" not in entities


def test_extract_entities_regex_empty_text():
    assert _extract_entities_regex("") == []


def test_extract_entities_regex_no_entities():
    # All lowercase, no proper nouns
    entities = _extract_entities_regex("the cat sat on the mat and ran away.")
    assert entities == []


# ── extract_regular（全量特征）─────────────────────────────────────────────

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


def test_extract_regular_finds_entities():
    """实体识别核心测试：不依赖 spaCy 也必须能识别到。"""
    feats = extract_regular("Michael went to Paris and met with John.")
    entities = feats.get("entities", [])
    assert len(entities) > 0, "entities must not be empty — regex fallback should work without spaCy"
    assert any(e in entities for e in ["Michael", "Paris", "John"])


def test_extract_regular_finds_pronouns():
    feats = extract_regular("He gave the book to her.")
    assert "he" in feats.get("pronouns", [])
    assert "her" in feats.get("pronouns", [])


# ── extract_llm ────────────────────────────────────────────────────────────

def test_extract_llm_returns_empty_when_no_engine():
    feats = extract_llm("Any text.", llm_engine=None)
    assert feats == {}


# ── count_method_features ──────────────────────────────────────────────────

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


def test_count_double_negation_attack_has_negation():
    feats = {"negations": ["not", "never"]}
    assert count_method_features(feats, "double_negation_attack") == 1


def test_count_double_negation_attack_no_negation():
    feats = {"negations": []}
    assert count_method_features(feats, "double_negation_attack") == 0


def test_count_scope_degree_scaling():
    feats = {"degree_words": ["all", "must"]}
    assert count_method_features(feats, "scope_degree_scaling") == 2


def test_count_logical_operator_rewrite():
    feats = {"logic_words": ["because", "however"]}
    assert count_method_features(feats, "logical_operator_rewrite") == 2


def test_count_temporal_causal_inversion():
    feats = {"sequence_words": ["before", "after"]}
    assert count_method_features(feats, "temporal_causal_inversion") == 2


def test_count_role_swap():
    feats = {"subject_candidates": ["dog"], "object_candidates": ["man", "ball"]}
    assert count_method_features(feats, "role_swap") == 1


def test_count_concept_hierarchy_shift():
    feats = {"entities": ["Michael", "Paris"]}
    assert count_method_features(feats, "concept_hierarchy_shift") == 2


def test_count_premise_disruption_always_one():
    assert count_method_features({}, "premise_disruption") == 1


def test_count_unknown_method_returns_zero():
    assert count_method_features({}, "nonexistent_method") == 0
