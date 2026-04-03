from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constructors import apply_method
from .formatter import METHOD_NAMES
from .llm_engine import LocalLLMEngine


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def ensure_text3_valid(text2: str, text3: Optional[str]) -> Optional[str]:
    if text3 is None:
        return None
    t3 = normalize_text(text3)
    t2 = normalize_text(text2)
    if not t3:
        return None
    if t3 == t2:
        return None
    return text3.strip()


def pick_default_methods(formatted_item: Dict[str, Any], max_methods: int = 3) -> List[str]:
    """
    Choose methods with availability flag=1.
    By default pick the earliest (fixed order) available method.
    """
    available = formatted_item.get("methods_available", {}) or {}
    chosen: List[str] = []
    for name in METHOD_NAMES:
        if available.get(name, 0) == 1:
            chosen.append(name)
        if len(chosen) >= max_methods:
            break
    return chosen


def generate_text3_for_item(
    formatted_item: Dict[str, Any],
    methods: Optional[Sequence[str]] = None,
    llm_engine: Optional[LocalLLMEngine] = None,
    max_attempts: int = 3,
) -> Tuple[Optional[str], List[str]]:
    """
    Apply one method or a sequence (1~3) to generate text3 from text2.
    """
    text2 = formatted_item.get("text2")
    if not isinstance(text2, str) or not text2.strip():
        return None, []

    features = formatted_item.get("features", {}) or {}
    if methods is None or len(methods) == 0:
        methods = pick_default_methods(formatted_item, max_methods=1)

    methods = list(methods)[:3]
    if not methods:
        return None, []

    # Try a few attempts (useful if some methods are non-deterministic via LLM).
    for _ in range(max_attempts):
        cur = text2
        applied_any = False
        applied_methods: List[str] = []
        for m in methods:
            out = apply_method(method_name=m, formatted_item=formatted_item, text=cur, features=features, llm_engine=llm_engine)
            out_valid = ensure_text3_valid(text2=text2, text3=out)
            if out_valid is None:
                continue
            cur = out_valid
            applied_any = True
            applied_methods.append(m)
        if applied_any:
            # Validate final result differs from original text2
            return ensure_text3_valid(text2=text2, text3=cur), applied_methods
    return None, []


def generate_dataset(
    formatted_data: Sequence[Dict[str, Any]],
    methods: Optional[Sequence[str]] = None,
    llm_engine: Optional[LocalLLMEngine] = None,
) -> List[Dict[str, Any]]:
    """
    Create final dataset rows: id, text1, text2, text3, score.
    """
    rows: List[Dict[str, Any]] = []
    for item in formatted_data:
        text3, methods_used = generate_text3_for_item(item, methods=methods, llm_engine=llm_engine)
        if text3 is None:
            continue
        rows.append(
            {
                "id": item.get("id"),
                "text1": item.get("text1"),
                "text2": item.get("text2"),
                "text3": text3,
                "score": item.get("score"),
                "methods_used": methods_used,
            }
        )
    return rows
