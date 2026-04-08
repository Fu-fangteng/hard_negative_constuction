"""
Stage 2 LLM Prompts
====================
包含两类 prompt：

1. 特征提取（Feature Extraction）
   - FEATURE_SYSTEM_PROMPT
   - build_feature_user_prompt(text) -> str

2. 困难负样本构造（Hard Negative Construction）
   - CONSTRUCTION_SYSTEM_PROMPT
   - build_construction_prompt(method_name, text) -> str
   每种方法对应一个独立的 prompt 模板，指导 LLM 生成特定类型的扰动。

设计原则：
  - 构造 prompt 只要求 LLM 输出修改后的句子，不需要解释
  - 特征提取 prompt 要求返回严格 JSON，字段与 Regular 路径完全一致
"""
from __future__ import annotations

import json
from typing import Dict


# ══════════════════════════════════════════════════════════════════════════════
# 一、特征提取 Prompts
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA: Dict[str, object] = {
    "entities":           ["named entities: person names, place names, organizations"],
    "numbers":            ["numeric tokens: integers, decimals, percentages, money"],
    "logic_words":        ["logical connectives: because, however, if, therefore, but, although, unless, so, while"],
    "sequence_words":     ["temporal/sequence markers: before, after, then, first, finally, later, previously"],
    "degree_words":       ["quantifiers and degree words: all, some, most, many, few, must, may, never, always"],
    "negations":          ["negation words: not, no, never, none, nothing, nobody, neither, nor, without"],
    "pronouns":           ["personal pronouns: he, she, they, it, him, her, them, his, her, their, its"],
    "subject_candidates": ["likely grammatical subjects (noun or pronoun doing the action)"],
    "object_candidates":  ["likely grammatical objects (noun or pronoun receiving the action)"],
}

FEATURE_SYSTEM_PROMPT = (
    "You are a precise information extraction engine for NLP data processing. "
    "Return strict JSON only. Do not output markdown, code blocks, or any explanation."
)


def build_feature_user_prompt(text: str) -> str:
    """构建特征提取的 user prompt。"""
    schema_text = json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "Extract lexical and semantic features from the sentence below.\n\n"
        "Rules:\n"
        "1. Output valid JSON only — no markdown, no extra text.\n"
        "2. Each field must be a list of surface-form strings copied exactly from the input.\n"
        "3. Return an empty list [] if a field is not found in the sentence.\n"
        "4. Do NOT invent tokens not present in the input.\n\n"
        f"JSON schema:\n{schema_text}\n\n"
        f"Sentence: {text}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 二、困难负样本构造 Prompts
# ══════════════════════════════════════════════════════════════════════════════

CONSTRUCTION_SYSTEM_PROMPT = (
    "You are a hard negative example generator for NLP contrastive learning. "
    "Your task is to make a small, targeted modification to a sentence as instructed. "
    "Output ONLY the modified sentence — no explanations, no quotes, no extra text."
)

# 每种方法的 user prompt 模板，{text} 为占位符
_CONSTRUCTION_TEMPLATES: Dict[str, str] = {

    "numeric_metric_transform": (
        "Task: Change one or more numbers in the sentence to different values.\n"
        "Rules:\n"
        "- Keep all non-numeric parts exactly the same.\n"
        "- The new number(s) must be noticeably different (not just ±0.1).\n"
        "- Preserve units (%, $, km, etc.) if present.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "entity_pronoun_substitution": (
        "Task: Replace a name or pronoun in the sentence with a different one.\n"
        "Rules:\n"
        "- If there is a gendered pronoun (he/she, him/her, his/her), swap it.\n"
        "- Otherwise, replace a person name, place name, or organization with a "
        "plausible alternative (e.g. John→Michael, Paris→London, Google→Apple).\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "scope_degree_scaling": (
        "Task: Change a quantifier or degree word in the sentence to alter its scope or intensity.\n"
        "Rules:\n"
        "- Replace one word from: all→some, most→few, always→sometimes, "
        "never→sometimes, must→might, very→slightly, completely→partially.\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "direct_negation_attack": (
        "Task: Negate the main claim of the sentence by inserting 'not'.\n"
        "Rules:\n"
        "- Add 'not' after the first auxiliary verb (is/are/was/were/can/will/do/does/did/has/have).\n"
        "- If no auxiliary verb exists, add 'does not' or 'did not' before the main verb.\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "double_negation_attack": (
        "Task: Remove an existing negation from the sentence to flip its polarity.\n"
        "Rules:\n"
        "- Remove one negation word: not, never, no, nobody, nothing, none, without.\n"
        "- If the sentence has a contraction like 'isn't', convert it to 'is'.\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "logical_operator_rewrite": (
        "Task: Replace a logical connective to change the logical relationship.\n"
        "Rules:\n"
        "- Use one of these substitutions: because→although, although→because, "
        "if→unless, unless→if, therefore→however, however→therefore, so→but, "
        "since→while, when→while, while→when.\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "role_swap": (
        "Task: Swap the subject and the object of the sentence.\n"
        "Rules:\n"
        "- Exchange who is doing the action and who is receiving it.\n"
        "- Keep the verb phrase exactly the same.\n"
        "- Example: 'The dog chased the cat.' → 'The cat chased the dog.'\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "temporal_causal_inversion": (
        "Task: Swap a temporal marker to reverse the order of events.\n"
        "Rules:\n"
        "- Use one of: before↔after, first↔last, first→finally, previously→later, "
        "later→previously, then→before.\n"
        "- Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "concept_hierarchy_shift": (
        "Task: Replace a specific noun with a more general category (hypernym) or "
        "a different category at the same level.\n"
        "Rules:\n"
        "- Examples: dog→animal, car→vehicle, apple→fruit, doctor→person, "
        "soccer→sport, piano→instrument.\n"
        "- Replace only one noun. Keep everything else exactly the same.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),

    "premise_disruption": (
        "Task: Add a contradictory prefix to the beginning of the sentence.\n"
        "Rules:\n"
        "- Prepend exactly one of these phrases:\n"
        "  'Contrary to what was stated, '\n"
        "  'Despite the opposite being true, '\n"
        "  'Although the evidence suggests otherwise, '\n"
        "- Lowercase the original sentence's first letter after the prefix.\n"
        "- Keep the rest of the sentence unchanged.\n\n"
        "Sentence: {text}\n\n"
        "Modified sentence:"
    ),
}


def build_construction_prompt(method_name: str, text: str) -> str:
    """
    根据方法名和输入句子，构建构造 prompt 的 user 部分。

    参数：
        method_name : 10 种方法之一
        text        : 待扰动的 positive 句子

    返回：
        格式化后的 user prompt 字符串

    异常：
        ValueError — method_name 不在已知方法列表中
    """
    template = _CONSTRUCTION_TEMPLATES.get(method_name)
    if template is None:
        raise ValueError(
            f"Unknown method: {method_name!r}. "
            f"Valid methods: {list(_CONSTRUCTION_TEMPLATES)}"
        )
    return template.format(text=text)
