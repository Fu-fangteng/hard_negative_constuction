from __future__ import annotations

import json
from typing import Dict


FEATURE_SCHEMA: Dict[str, object] = {
    "entities": ["person/org/location/pronoun mentions"],
    "numbers": ["cardinal/percent/money/time/measurement"],
    "logic_words": ["because/however/if/therefore/etc."],
    "sequence_words": ["before/after/then/finally/etc."],
    "degree_words": ["all/some/most/must/maybe/never/etc."],
    "negations": ["not/no/never/nothing/none"],
    "subject_candidates": ["possible grammatical subject spans"],
    "object_candidates": ["possible grammatical object spans"],
}


FEATURE_SYSTEM_PROMPT = (
    "You are an information extraction engine for semantic textual similarity data. "
    "Return strict JSON only. Do not output markdown or explanations."
)


def build_feature_user_prompt(text: str) -> str:
    """
    Prompt used by formatter to extract editable features from text2.
    """
    schema_text = json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "Task: Extract lexical/semantic features from the given sentence for hard-negative "
        "construction.\n\n"
        "Requirements:\n"
        "1) Output valid JSON only.\n"
        "2) Keep each field as a list of surface spans from the input text.\n"
        "3) If a field is unavailable, return an empty list.\n"
        "4) Do not hallucinate tokens not present in the input.\n\n"
        f"Target JSON schema:\n{schema_text}\n\n"
        f"Input sentence:\n{text}\n"
    )
