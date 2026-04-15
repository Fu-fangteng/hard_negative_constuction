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
Make the sentence factually wrong by changing one number to a clearly different value.
The rest of the sentence must stay exactly the same — only the number changes.

Examples:
- "He ran 5 miles every day." → "He ran 12 miles every day."
- "About 40% of voters supported the bill." → "About 85% of voters supported the bill."
- "The bridge was built in 1920." → "The bridge was built in 1987."
- "She scored 95 on the test." → "She scored 42 on the test."

Now rewrite (output only the result):
{text}""",

    "entity_pronoun_substitution": """\
Make the sentence describe a different participant by swapping one pronoun or one proper name for another.
The sentence structure and all other words must stay exactly the same.

Examples:
- "She gave him the keys." → "He gave her the keys."
- "John won the award." → "Michael won the award."
- "Google acquired the startup." → "Apple acquired the startup."
- "They sent it to her." → "They sent it to him."

Now rewrite (output only the result):
{text}""",

    "scope_degree_scaling": """\
Change how broadly or strongly the sentence's claim applies by replacing one quantifier or modal word with one that has a clearly different strength.
The rest of the sentence must stay exactly the same.

Good substitution pairs: all↔few, always↔rarely, must↔may, many↔few, most↔some, never↔often.

Examples:
- "All students passed the exam." → "Few students passed the exam."
- "She always arrives on time." → "She rarely arrives on time."
- "The medicine must be taken daily." → "The medicine may be taken daily."
- "Most residents supported the plan." → "Few residents supported the plan."

Now rewrite (output only the result):
{text}""",

    "direct_negation_attack": """\
Flip the truth value of the sentence's main claim.
- If the sentence is affirmative: insert "not" after the first auxiliary verb (is/are/was/were/can/could/will/would/has/have/had/should/must/may/might/do/does/did), or add "does not" / "did not" before the main verb if there is no auxiliary.
- If the sentence is already negative (contains not/no/never/n't): remove the negation word to make it affirmative.

Examples:
- "She is running fast." → "She is not running fast."
- "They can swim well." → "They cannot swim well."
- "He studies medicine." → "He does not study medicine."
- "There is no shooting in the scene." → "There is shooting in the scene."
- "The report was never published." → "The report was published."

Now rewrite (output only the result):
{text}""",

    "double_negation_attack": """\
Turn a negative sentence into a positive one by removing its negation.
Find the negation word (not, no, never, n't contraction) and remove it so the sentence now asserts the opposite.
Keep all other words exactly the same.

Examples:
- "She is not happy." → "She is happy."
- "They don't like vegetables." → "They like vegetables."
- "He never goes to the gym." → "He goes to the gym."
- "There was no evidence of fraud." → "There was evidence of fraud."

Now rewrite (output only the result):
{text}""",

    "logical_operator_rewrite": """\
Change the logical relationship between two parts of the sentence by swapping one connective for one with a different meaning.
Use pairs that change the relationship: because↔although, if↔unless, therefore↔however, so↔but, since↔while, when↔while.
Keep all other words exactly the same.

Examples:
- "She passed because she studied hard." → "She passed although she studied hard."
- "He left early so he could catch the train." → "He left early but he could catch the train."
- "I stayed home because it was raining." → "I stayed home although it was raining."
- "Call me if you need help." → "Call me unless you need help."

Now rewrite (output only the result):
{text}""",

    "role_swap": """\
Reverse who does the action and who receives it — swap the subject and the object.
Keep the verb, adjectives, and all other words exactly the same; only the two main participants switch places.

Examples:
- "The dog chased the cat." → "The cat chased the dog."
- "Mary helped John." → "John helped Mary."
- "The police arrested the suspect." → "The suspect arrested the police."
- "The teacher praised the student." → "The student praised the teacher."

Now rewrite (output only the result):
{text}""",

    "temporal_causal_inversion": """\
Reverse the order of events by changing one temporal marker to its opposite.
Use pairs: before↔after, first↔last, previously↔later, finally↔first.
Keep all other words exactly the same.

Examples:
- "She ate dinner before watching TV." → "She ate dinner after watching TV."
- "He finished the report before the meeting." → "He finished the report after the meeting."
- "First, mix the ingredients, then bake." → "Last, mix the ingredients, then bake."
- "Previously she had worked in finance." → "Later she had worked in finance."

Now rewrite (output only the result):
{text}""",

    "concept_hierarchy_shift": """\
Replace one specific noun with a different but plausible word — either a broader category (hypernym) or a different concept at the same level — so the sentence remains grammatical but describes something different.
Keep all other words exactly the same.

Examples:
- "The dog ran across the yard." → "The animal ran across the yard."
- "She drives a car to work." → "She drives a truck to work."
- "He bought an apple at the store." → "He bought a mango at the store."
- "The doctor examined the patient." → "The nurse examined the patient."

Now rewrite (output only the result):
{text}""",

    "premise_disruption": """\
Add a short contradictory phrase at the very beginning of the sentence to signal that its content conflicts with known facts.
Use one of these openings: "Contrary to what was stated, " / "Despite the opposite being true, " / "Although the evidence suggests otherwise, " / "In theory, ".
Lowercase the first letter of the original sentence after the prefix.

Examples:
- "The team won the championship." → "Contrary to what was stated, the team won the championship."
- "Vaccines are effective." → "Despite the opposite being true, vaccines are effective."
- "The economy is growing." → "Although the evidence suggests otherwise, the economy is growing."
- "Exercise improves mental health." → "In theory, exercise improves mental health."

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
