"""
Stage 2 Constructors
====================
直接复用 Stage 1 的全部 10 种构造方法，并对
entity_pronoun_substitution 进行增强：
  - 保留 Stage 1 的代词替换逻辑（he↔she 等）
  - 新增真实名词替换表（人名/地名/机构），
    而非 Stage 1 的 "another <entity>" 兜底策略

其余 9 种方法完全透传 Stage 1 实现。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 直接导入 Stage 1 已验证的全部构造函数 ──────────────────────────────────
from stage1.constructors import (
    apply_method as _stage1_apply_method,
    numeric_metric_transform,
    scope_degree_scaling,
    direct_negation_attack,
    double_negation_attack,
    logical_operator_rewrite,
    role_swap,
    temporal_causal_inversion,
    concept_hierarchy_shift,
    premise_disruption,
    _replace_pronoun,          # Stage 1 内部代词替换，直接复用
)
from stage2.prompts import CONSTRUCTION_SYSTEM_PROMPT, build_construction_prompt

# ── 实体替换表（人名 / 地名 / 机构）────────────────────────────────────────

_PERSON_MAP: Dict[str, str] = {
    # 男性名
    "john": "michael", "michael": "david", "david": "james",
    "james": "robert", "robert": "william", "william": "thomas",
    "thomas": "charles", "charles": "joseph", "joseph": "george",
    "george": "henry", "henry": "edward", "edward": "peter",
    "peter": "paul", "paul": "mark", "mark": "andrew",
    "tom": "jack", "jack": "harry", "harry": "oliver",
    "oliver": "noah", "noah": "liam", "bob": "charlie",
    "charlie": "dave", "dave": "frank", "frank": "steve",
    "steve": "kevin", "kevin": "brian", "brian": "scott",
    # 女性名
    "mary": "sarah", "sarah": "emma", "emma": "olivia",
    "olivia": "sophie", "sophie": "claire", "claire": "alice",
    "alice": "betty", "betty": "helen", "helen": "jane",
    "jane": "anne", "anne": "kate", "kate": "laura",
    "laura": "lucy", "lucy": "grace", "grace": "rose",
}

_PLACE_MAP: Dict[str, str] = {
    # 城市
    "paris": "london", "london": "berlin", "berlin": "tokyo",
    "tokyo": "rome", "rome": "madrid", "madrid": "vienna",
    "vienna": "amsterdam", "amsterdam": "brussels", "brussels": "zurich",
    "new york": "los angeles", "los angeles": "chicago",
    "chicago": "houston", "houston": "phoenix", "phoenix": "seattle",
    "seattle": "boston", "boston": "miami", "miami": "denver",
    "beijing": "shanghai", "shanghai": "guangzhou",
    "moscow": "st. petersburg", "sydney": "melbourne",
    "toronto": "vancouver", "montreal": "calgary",
    # 国家
    "france": "germany", "germany": "japan", "japan": "china",
    "china": "india", "india": "brazil", "brazil": "russia",
    "russia": "canada", "canada": "australia", "australia": "mexico",
    "mexico": "italy", "italy": "spain", "spain": "south korea",
    "america": "europe", "europe": "asia",
    "united states": "united kingdom",
    "united kingdom": "united states",
    # 州/省
    "california": "texas", "texas": "florida", "florida": "new york",
    "new york": "illinois", "illinois": "ohio",
}

_ORG_MAP: Dict[str, str] = {
    # 科技公司
    "google": "apple", "apple": "microsoft", "microsoft": "amazon",
    "amazon": "facebook", "facebook": "twitter", "twitter": "netflix",
    "netflix": "tesla", "tesla": "uber", "uber": "airbnb",
    "airbnb": "spotify", "spotify": "adobe", "adobe": "oracle",
    "oracle": "ibm", "ibm": "intel", "intel": "nvidia",
    # 大学
    "harvard": "yale", "yale": "princeton", "princeton": "stanford",
    "stanford": "mit", "mit": "caltech", "caltech": "columbia",
    "oxford": "cambridge", "cambridge": "oxford",
    # 机构缩写
    "nasa": "esa", "esa": "nato", "nato": "un",
    "un": "eu", "eu": "who", "who": "imf",
    "cia": "fbi", "fbi": "nsa", "nsa": "cia",
    "gop": "dnc", "dnc": "gop",
}

# 合并所有替换表（全小写键 → 替换词）
_ENTITY_REPLACE_MAP: Dict[str, str] = {**_PERSON_MAP, **_PLACE_MAP, **_ORG_MAP}


def _entity_swap(text: str, entities: List[str]) -> Optional[str]:
    """
    用替换表交换第一个被识别到且有对应替换词的实体。
    保留原始大小写风格（全大写 / 首字母大写 / 小写）。
    """
    for ent in entities:
        replacement = _ENTITY_REPLACE_MAP.get(ent.lower())
        if replacement is None:
            continue
        pat = re.compile(rf"\b{re.escape(ent)}\b", re.IGNORECASE)
        m = pat.search(text)
        if not m:
            continue
        found = m.group(0)
        if found.isupper():
            repl = replacement.upper()
        elif found[0].isupper():
            # 首字母大写（多词时每词首字母大写）
            repl = " ".join(w.capitalize() for w in replacement.split())
        else:
            repl = replacement.lower()
        new_text = pat.sub(repl, text, count=1)
        if new_text != text:
            return new_text
    return None


def entity_pronoun_substitution(
    formatted_item: Dict[str, Any],
    text: str,
    features: Dict[str, Any],
    llm_engine: Optional[LocalLLMEngine] = None,
) -> Optional[str]:
    """
    增强版实体/代词替换：
    1. 优先做代词替换（he↔she 等，来自 Stage 1）
    2. 其次用替换表换实体（人名/地名/机构）
    """
    # Step 1: pronoun swap（直接复用 Stage 1 逻辑）
    out = _replace_pronoun(text)
    if out is not None:
        return out

    # Step 2: entity swap via lookup table
    entities = features.get("entities") or []
    out = _entity_swap(text, entities)
    if out is not None:
        return out

    return None


# ── LLM 直接构造（LLM 路径专用）─────────────────────────────────────────

def _parse_llm_output(raw: str, original: str) -> Optional[str]:
    """
    解析并验证 LLM 构造输出。
    过滤以下几类无效输出：
      1. 空输出
      2. 与原文相同
      3. 句末孤立的 "not"  — 语法错误（模型找不到助动词时的 fallback）
      4. 仅将 "no" 改为 "not" — 语义几乎等价，不构成困难负样本
    """
    if not raw:
        return None
    out = raw.strip().strip('"').strip("'").strip()
    # 去除模型可能回显的 prompt 标签
    for prefix in (
        "Modified sentence:", "Answer:", "Output:", "Result:",
        "Modified:", "Sentence:",
    ):
        if out.lower().startswith(prefix.lower()):
            out = out[len(prefix):].strip()
    out = out.strip().strip('"').strip("'")
    if not out:
        return None

    # ── 规则 1：与原文相同（规范化空白+大小写）─────────────────────────────
    if " ".join(out.lower().split()) == " ".join(original.lower().split()):
        return None

    # ── 规则 2：句末孤立 "not" — 语法错误 ─────────────────────────────────
    # "...man not."  /  "...man not"  → 无效
    last_word = re.sub(r"[^\w]", "", out.split()[-1]).lower() if out.split() else ""
    if last_word == "not":
        return None

    # ── 规则 3：仅把 "no" 换成 "not" — 语义等价 ───────────────────────────
    # e.g. "no shooting" → "not shooting"：两者均表示否定，语义高度相似
    norm_orig = re.sub(r"\bno\b", "not", original, flags=re.IGNORECASE)
    if " ".join(norm_orig.lower().split()) == " ".join(out.lower().split()):
        return None

    return out


def _apply_llm_construction(method_name: str, text: str, llm_engine) -> Optional[str]:
    """
    调用 Qwen3 直接生成困难负样本。

    - 成功：返回修改后的句子
    - 无法修改（与原文相同或空）：返回 None
    - 异常：向上抛出，由 builder.py 统一捕获记录（failure_reason="exception"）
    """
    try:
        user_prompt = build_construction_prompt(method_name, text)
    except ValueError:
        return None
    # 不在此处捕获异常——让 builder.py 的外层 try-except 记录详细错误
    raw = llm_engine.generate(
        system_prompt=CONSTRUCTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return _parse_llm_output(raw, text)


# ── 统一调度入口 ──────────────────────────────────────────────────────────

_METHOD_REGISTRY = {
    "numeric_metric_transform":    numeric_metric_transform,
    "entity_pronoun_substitution": entity_pronoun_substitution,   # 增强版
    "scope_degree_scaling":        scope_degree_scaling,
    "direct_negation_attack":      direct_negation_attack,
    "double_negation_attack":      double_negation_attack,
    "logical_operator_rewrite":    logical_operator_rewrite,
    "role_swap":                   role_swap,
    "temporal_causal_inversion":   temporal_causal_inversion,
    "concept_hierarchy_shift":     concept_hierarchy_shift,
    "premise_disruption":          premise_disruption,
}


def apply_method(
    method_name: str,
    formatted_item: Dict[str, Any],
    text: str,
    features: Dict[str, Any],
    llm_engine: Optional[Any] = None,
) -> Optional[str]:
    """
    构造调度入口。

    - LLM 路径（llm_engine 非 None 且 ready）：
        直接调用 Qwen3 生成扰动句，不使用规则。
        失败（输出为空/与原文相同）时返回 None——不做 fallback。
    - Regular 路径（llm_engine 为 None）：
        完全走规则逻辑。
    """
    if llm_engine is not None and getattr(llm_engine, "ready", False):
        return _apply_llm_construction(method_name, text, llm_engine)

    fn = _METHOD_REGISTRY.get(method_name)
    if fn is None:
        raise ValueError(f"Unknown method: {method_name!r}")
    return fn(formatted_item=formatted_item, text=text, features=features)
