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

设计原则：
  - 每种方法提供 2-3 个示例，引导小模型（1.7B）理解任务
  - 明确说明"只输出修改后的句子"，不要解释
  - 不在用户消息末尾加 "Modified sentence:" 等补全式提示
    （chat 模型不需要补全提示，示例已经足够引导输出格式）
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
    "You are a sentence rewriting assistant. "
    "Follow the instruction exactly and output only the rewritten sentence. "
    "Do not explain, do not add any other text."
)

_CONSTRUCTION_TEMPLATES: Dict[str, str] = {

    "numeric_metric_transform": """\
Rewrite the sentence by changing one or more numbers to different values.
Keep all non-numeric words exactly the same.

Examples:
- "The temperature rose from 20°C to 30°C." → "The temperature rose from 35°C to 50°C."
- "He ran 5 miles every day." → "He ran 2 miles every day."
- "About 40% of voters supported the bill." → "About 75% of voters supported the bill."

Now rewrite (output only the result):
{text}""",

    "entity_pronoun_substitution": """\
Rewrite the sentence by replacing one name or pronoun with a different one.
Keep everything else exactly the same.

Examples:
- "John went to Paris." → "Michael went to London."
- "She gave him the book." → "He gave her the book."
- "Google acquired the startup." → "Apple acquired the startup."

Now rewrite (output only the result):
{text}""",

    "scope_degree_scaling": """\
Rewrite the sentence by changing one quantifier or degree word to its opposite or a weaker/stronger alternative.
Keep everything else exactly the same.

Examples:
- "All students passed the exam." → "Few students passed the exam."
- "She always arrives on time." → "She rarely arrives on time."
- "The medicine must be taken daily." → "The medicine may be taken daily."

Now rewrite (output only the result):
{text}""",

    "direct_negation_attack": """\
Rewrite the sentence to negate its main claim by inserting "not" after the first auxiliary verb.
If there is no auxiliary verb, add "does not" or "did not" before the main verb.
Keep everything else exactly the same.

Examples:
- "She is running fast." → "She is not running fast."
- "They can swim well." → "They cannot swim well."
- "He studies medicine." → "He does not study medicine."
- "The dog chased the cat." → "The dog did not chase the cat."

Now rewrite (output only the result):
{text}""",

    "double_negation_attack": """\
Rewrite the sentence by removing one negation word to flip its polarity.
If the sentence has a contraction like "isn't" or "don't", expand and remove the "not".
Keep everything else exactly the same.

Examples:
- "She is not happy." → "She is happy."
- "They don't like vegetables." → "They like vegetables."
- "He never goes to the gym." → "He goes to the gym."

Now rewrite (output only the result):
{text}""",

    "logical_operator_rewrite": """\
Rewrite the sentence by replacing one logical connective with a different one that changes the logical relationship.
Use these substitutions: because↔although, if↔unless, therefore↔however, so↔but, since↔while.
Keep everything else exactly the same.

Examples:
- "She passed because she studied hard." → "She passed although she studied hard."
- "I stayed home because it was raining." → "I stayed home although it was raining."
- "He left early so he could catch the train." → "He left early but he could catch the train."

Now rewrite (output only the result):
{text}""",

    "role_swap": """\
Rewrite the sentence by swapping the subject and the object (who does what to whom).
Keep the verb and all other words exactly the same.

Examples:
- "The dog chased the cat." → "The cat chased the dog."
- "Mary helped John." → "John helped Mary."
- "The police arrested the suspect." → "The suspect arrested the police."

Now rewrite (output only the result):
{text}""",

    "temporal_causal_inversion": """\
Rewrite the sentence by swapping one temporal marker to reverse the order of events.
Use these substitutions: before↔after, first↔last, previously↔later, finally↔first.
Keep everything else exactly the same.

Examples:
- "She ate dinner before watching TV." → "She ate dinner after watching TV."
- "First mix the ingredients, then bake." → "Last mix the ingredients, then bake."
- "He finished the report before the meeting." → "He finished the report after the meeting."

Now rewrite (output only the result):
{text}""",

    "concept_hierarchy_shift": """\
Rewrite the sentence by replacing one specific noun with a more general category word (hypernym) or a different category.
Keep everything else exactly the same.

Examples:
- "The dog ran across the yard." → "The animal ran across the yard."
- "She drives a car to work." → "She drives a vehicle to work."
- "He bought an apple at the store." → "He bought a fruit at the store."

Now rewrite (output only the result):
{text}""",

    "premise_disruption": """\
Rewrite the sentence by adding a short contradictory prefix at the beginning.
Choose one of these prefixes: "Contrary to what was stated, " / "Despite the opposite being true, " / "Although the evidence suggests otherwise, ".
Lowercase the first letter of the original sentence after the prefix.

Examples:
- "The team won the championship." → "Contrary to what was stated, the team won the championship."
- "Vaccines are effective." → "Despite the opposite being true, vaccines are effective."
- "The economy is growing." → "Although the evidence suggests otherwise, the economy is growing."

Now rewrite (output only the result):
{text}""",
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
