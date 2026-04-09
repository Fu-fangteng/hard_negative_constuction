"""
Stage 2 Feature Extractor
=========================
直接复用 Stage 1 经过验证的特征提取逻辑（formatter.py），
并在此基础上补充一个**不依赖 spaCy** 的正则实体识别，
确保即使 spaCy 未安装，entities 字段也能有效填充。

Regular 路径提取顺序：
  1. _regex_extract()         —— 数字/逻辑词/否定词/代词等（Stage 1 原版）
  2. _extract_entities_regex() —— 新增：正则启发式实体识别（大写专有名词）
  3. _safe_spacy_extract()    —— 可选：spaCy NER（有则补充，无则跳过）
  以上三路结果 _merge_features() 合并去重

LLM 路径同 Stage 1。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 引入 Stage 1 已验证的提取函数 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.formatter import (
    _merge_features,
    _parse_llm_json,
    _regex_extract,
    _safe_spacy_extract,
)
from stage2.prompts import FEATURE_SYSTEM_PROMPT, build_feature_user_prompt

# ── 正则实体识别所需常量 ────────────────────────────────────────────────────

# 这些词即使大写也不是专有名词（冠词、代词、连词、介词、星期、月份等）
# 专有名词不在此列——即使出现在句首也应被识别（如 "Michael went to..."）
_COMMON_CAPS: frozenset = frozenset({
    # 冠词
    "a", "an", "the",
    # 人称代词（主格，常出现在句首）
    "i", "he", "she", "they", "it", "we", "you",
    # 指示代词 / 副词
    "this", "that", "these", "those", "there", "here",
    # 连词 / 连接副词
    "and", "or", "but", "so", "yet", "nor",
    "however", "therefore", "although", "because", "when", "if",
    "while", "unless", "then", "than", "as",
    # 介词
    "in", "on", "at", "to", "of", "for", "by", "from", "with",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "among", "about", "against", "along", "around",
    # 助动词 / 系动词
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "shall",
    # 否定词
    "not", "no", "nor", "neither",
    # 量词（通常不是专有名词）
    "all", "some", "most", "many", "few", "each", "every", "any", "several",
    # 星期
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # 月份
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
})

# 称谓前缀（后接大写专有名词则整体作为实体）
_TITLE_PAT = re.compile(
    r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr|Gen|Sgt|Cpl|Lt|Capt|Maj|Col|Rev|Pres|Gov)\.?\s+'
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
)
# 连续大写开头的单词序列（专有名词短语，如 "New York", "John Smith"）
_CAP_SEQ_PAT = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b')
# 全大写缩写（NATO, NASA, WHO, EU …）
_ACRONYM_PAT = re.compile(r'\b([A-Z]{2,5})\b')


def _extract_entities_regex(text: str) -> List[str]:
    """
    不依赖 spaCy 的启发式实体识别。

    策略：
    1. 称谓 + 姓名模式     e.g.  "Dr. Smith", "Prof. Alan Turing"
    2. 句中大写词序列      e.g.  "London", "New York", "Marie Curie"
       （跳过句首第一个词，避免把普通句首大写误判）
    3. 全大写缩写          e.g.  "NASA", "WHO", "EU"
    """
    entities: List[str] = []

    # 1. 称谓 + 姓名（最高可信度，不需要位置过滤）
    for m in _TITLE_PAT.finditer(text):
        entities.append(m.group(0).strip())

    # 2. 大写词序列（专有名词即使在句首也应捕获，如 "Michael went to..."）
    #    只用词表过滤掉常见的冠词/代词/连词等，不做位置过滤
    for m in _CAP_SEQ_PAT.finditer(text):
        word = m.group(1)
        if word.lower() in _COMMON_CAPS:
            continue          # 过滤非专有名词的常见大写词
        entities.append(word)

    # 3. 全大写缩写（如 NASA、WHO、EU；排除纯数字序列）
    for m in _ACRONYM_PAT.finditer(text):
        acronym = m.group(1)
        if not any(c.isdigit() for c in acronym):
            entities.append(acronym)

    return sorted(set(entities))


# ── 对外接口 ──────────────────────────────────────────────────────────────

def extract_regular(text: str) -> Dict[str, List[str]]:
    """
    Regular 路径：
      Stage1 regex（数字/逻辑词/否定词/代词）
      + 本模块正则实体识别（保证 entities 不为空）
      + spaCy（可选，有则补充更精确的 entities/subjects/objects）
    """
    regex_feats   = _regex_extract(text)
    entity_feats  = {"entities": _extract_entities_regex(text)}
    spacy_feats   = _safe_spacy_extract(text)          # 失败时返回 {}
    return _merge_features(regex_feats, entity_feats, spacy_feats)


def extract_llm(text: str, llm_engine: Optional[Any]) -> Dict[str, List[str]]:
    """
    LLM 特征提取路径（使用 Qwen3 或任何实现了 .generate() 接口的引擎）。
    llm_engine=None 或未加载时返回 {}。
    """
    if llm_engine is None or not getattr(llm_engine, "ready", False):
        return {}
    try:
        response = llm_engine.generate(
            system_prompt=FEATURE_SYSTEM_PROMPT,
            user_prompt=build_feature_user_prompt(text),
        )
    except Exception:
        return {}

    parsed = _parse_llm_json(response)
    if not isinstance(parsed, dict):
        return {}

    normalized: Dict[str, List[str]] = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            normalized[key] = [str(v).strip() for v in value if str(v).strip()]
    return normalized


def count_method_features(features: Dict[str, Any], method_name: str) -> int:
    """
    返回指定方法在当前特征下的可用特征数量。
    用于填充 dataset_methods_stat.json。
    """
    f = features
    mapping: Dict[str, int] = {
        "numeric_metric_transform":    len(f.get("numbers", [])),
        "entity_pronoun_substitution": len(f.get("entities", [])) + len(f.get("pronouns", [])),
        "scope_degree_scaling":        len(f.get("degree_words", [])),
        "direct_negation_attack":      0 if f.get("negations") else 1,
        "double_negation_attack":      1 if f.get("negations") else 0,
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
