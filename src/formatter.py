from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .data_utils import STSRecord
from .llm_engine import LocalLLMEngine
from .prompts import FEATURE_SYSTEM_PROMPT, build_feature_user_prompt

METHOD_NAMES: List[str] = [
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

LOGIC_WORDS = {
    "because",
    "therefore",
    "however",
    "but",
    "although",
    "if",
    "unless",
    "so",
    "while",
}
SEQUENCE_WORDS = {"before", "after", "then", "first", "finally", "later", "previously"}
DEGREE_WORDS = {
    "all",
    "some",
    "most",
    "many",
    "few",
    "must",
    "may",
    "might",
    "possibly",
    "never",
    "always",
}
NEGATIONS = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "without"}
PRONOUNS = {"he", "she", "they", "it", "him", "her", "them", "his", "their", "its"}


def _safe_spacy_extract(text: str) -> Dict[str, List[str]]:
    try:
        import spacy
    except Exception:
        return {"entities": [], "subject_candidates": [], "object_candidates": []}

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return {"entities": [], "subject_candidates": [], "object_candidates": []}

    doc = nlp(text)
    entities = sorted({ent.text for ent in doc.ents if ent.text.strip()})
    subjects = sorted({tok.text for tok in doc if tok.dep_ in {"nsubj", "nsubjpass", "csubj"}})
    objects = sorted({tok.text for tok in doc if tok.dep_ in {"dobj", "pobj", "obj", "iobj"}})
    return {
        "entities": entities,
        "subject_candidates": subjects,
        "object_candidates": objects,
    }


def _regex_extract(text: str) -> Dict[str, List[str]]:
    tokens = re.findall(r"\b[\w%$£€.-]+\b", text.lower())
    numbers = re.findall(r"[$£€]?\d+(?:,\d{3})*(?:\.\d+)?%?", text)
    logic_words = sorted({tok for tok in tokens if tok in LOGIC_WORDS})
    sequence_words = sorted({tok for tok in tokens if tok in SEQUENCE_WORDS})
    degree_words = sorted({tok for tok in tokens if tok in DEGREE_WORDS})
    negations = sorted({tok for tok in tokens if tok in NEGATIONS})
    pronouns = sorted({tok for tok in tokens if tok in PRONOUNS})
    return {
        "numbers": sorted(set(numbers)),
        "logic_words": logic_words,
        "sequence_words": sequence_words,
        "degree_words": degree_words,
        "negations": negations,
        "pronouns": pronouns,
    }


def _parse_llm_json(raw_output: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_output[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _llm_extract(text: str, llm_engine: Optional[LocalLLMEngine]) -> Dict[str, List[str]]:
    if llm_engine is None or not llm_engine.ready:
        return {}
    response = llm_engine.generate(
        system_prompt=FEATURE_SYSTEM_PROMPT,
        user_prompt=build_feature_user_prompt(text),
    )
    parsed = _parse_llm_json(response)
    if not isinstance(parsed, dict):
        return {}

    # Normalize unknown values into list[str].
    normalized: Dict[str, List[str]] = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            normalized[key] = [str(v).strip() for v in value if str(v).strip()]
    return normalized


def _merge_features(*feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for group in feature_groups:
        for key, values in group.items():
            bucket = merged.setdefault(key, [])
            bucket.extend(values)
    return {k: sorted(set(v)) for k, v in merged.items()}


def _method_availability(features: Dict[str, List[str]]) -> Dict[str, int]:
    numbers = len(features.get("numbers", [])) > 0
    entities = len(features.get("entities", [])) > 0 or len(features.get("pronouns", [])) > 0
    degree = len(features.get("degree_words", [])) > 0
    neg = len(features.get("negations", [])) > 0
    logic = len(features.get("logic_words", [])) > 0
    role = len(features.get("subject_candidates", [])) > 0 and len(features.get("object_candidates", [])) > 0
    temporal_or_logic = len(features.get("sequence_words", [])) > 0 or logic

    mapping = {
        "numeric_metric_transform": int(numbers),
        "entity_pronoun_substitution": int(entities),
        "scope_degree_scaling": int(degree),
        # Only applicable when the sentence doesn't already contain negation tokens.
        "direct_negation_attack": int(not neg),
        "double_negation_attack": int(neg),
        "logical_operator_rewrite": int(logic),
        "role_swap": int(role),
        "temporal_causal_inversion": int(temporal_or_logic),
        "concept_hierarchy_shift": int(entities),
        "premise_disruption": 1,
    }
    return mapping


def format_record(record: STSRecord, llm_engine: Optional[LocalLLMEngine] = None) -> Dict[str, Any]:
    text = record.text2
    features = _merge_features(
        _regex_extract(text),
        _safe_spacy_extract(text),
        _llm_extract(text, llm_engine),
    )
    methods = _method_availability(features)
    return {
        "id": record.id,
        "text1": record.text1,
        "text2": record.text2,
        "score": record.score,
        "features": features,
        "methods_available": methods,
    }


def format_dataset(
    records: Iterable[STSRecord],
    llm_engine: Optional[LocalLLMEngine] = None,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for record in records:
        formatted.append(format_record(record, llm_engine=llm_engine))
    return formatted


def build_methods_stat(formatted_data: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for item in formatted_data:
        methods = item.get("methods_available", {})
        feature_count = sum(len(v) for v in item.get("features", {}).values() if isinstance(v, list))
        row: Dict[str, Any] = {"id": item["id"], "feature_count": feature_count}
        for method in METHOD_NAMES:
            row[method] = int(methods.get(method, 0))
        stats.append(row)
    return stats


def export_formatter_outputs(
    formatted_data: List[Dict[str, Any]],
    methods_stat: List[Dict[str, Any]],
    formatted_data_path: str | Path,
    methods_stat_path: str | Path,
) -> None:
    fd_path = Path(formatted_data_path)
    ms_path = Path(methods_stat_path)
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    ms_path.parent.mkdir(parents=True, exist_ok=True)

    with fd_path.open("w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    with ms_path.open("w", encoding="utf-8") as f:
        json.dump(methods_stat, f, ensure_ascii=False, indent=2)
