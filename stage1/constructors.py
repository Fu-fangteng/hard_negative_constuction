from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm_engine import LocalLLMEngine


_NUMBER_PATTERN = re.compile(r"(?<!\w)([$£€]?\d+(?:,\d{3})*(?:\.\d+)?%?)(?!\w)", re.IGNORECASE)

# ── Irregular past-tense → base-form lookup ──────────────────────────────────
_IRR_PAST_TO_BASE: Dict[str, str] = {
    "ate": "eat", "became": "become", "began": "begin", "blew": "blow",
    "broke": "break", "brought": "bring", "built": "build", "bought": "buy",
    "came": "come", "caught": "catch", "chose": "choose", "cut": "cut",
    "dealt": "deal", "drew": "draw", "drove": "drive", "fell": "fall",
    "felt": "feel", "flew": "fly", "forgot": "forget", "found": "find",
    "froze": "freeze", "gave": "give", "got": "get", "grew": "grow",
    "held": "hold", "heard": "hear", "hit": "hit", "hurt": "hurt",
    "kept": "keep", "knew": "know", "knelt": "kneel", "led": "lead",
    "leapt": "leap", "left": "leave", "let": "let", "lost": "lose",
    "made": "make", "meant": "mean", "met": "meet", "paid": "pay",
    "put": "put", "ran": "run", "read": "read", "rode": "ride",
    "rose": "rise", "said": "say", "sat": "sit", "saw": "see",
    "sent": "send", "set": "set", "shot": "shoot", "slept": "sleep",
    "slid": "slide", "spoke": "speak", "spent": "spend", "stood": "stand",
    "stole": "steal", "swam": "swim", "swung": "swing", "swept": "sweep",
    "took": "take", "taught": "teach", "told": "tell", "thought": "think",
    "threw": "throw", "understood": "understand", "woke": "wake",
    "wore": "wear", "wept": "weep", "won": "win", "wrote": "write",
    "went": "go",
}

# ── spaCy module-level singleton (lazy-loaded, None = untried, False = unavailable)
_nlp_instance = None


def _get_nlp():
    global _nlp_instance
    if _nlp_instance is not None:
        return _nlp_instance
    try:
        import spacy
        _nlp_instance = spacy.load("en_core_web_sm")
    except Exception:
        _nlp_instance = False
    return _nlp_instance


def _strip_ed(w: str) -> str:
    """Strip -ed suffix → approximate base form for regular past-tense verbs."""
    if w.endswith("ied"):                                       # tried → try
        return w[:-3] + "y"
    if len(w) >= 5 and w[-3] == w[-4] and w[-3] not in "aeiou":
        return w[:-3]                                           # stopped → stop
    # VCeD pattern: vowel + consonant + e + d → remove 'd' only (leased→lease, moved→move)
    if (len(w) >= 4 and w[-1] == "d" and w[-2] == "e"
            and w[-3] not in "aeiou" and len(w) >= 5 and w[-4] in "aeiou"):
        return w[:-1]
    return w[:-2]                                               # walked→walk, shocked→shock


def _strip_s(w: str) -> str:
    """Strip 3rd-person-singular -s/-es → approximate base form."""
    if w.endswith("ies"):                       # tries → try
        return w[:-3] + "y"
    if w.endswith("oes"):                       # goes → go
        return w[:-2]
    if w.endswith("es") and len(w) > 3:
        stem = w[:-2]
        if stem.endswith(("s", "sh", "ch", "x", "z")):
            return stem                         # passes→pass, reaches→reach
        return w[:-1]                           # makes→make, takes→take
    return w[:-1]                               # eats→eat, walks→walk


def _negate_without_aux(text: str) -> Optional[str]:
    """
    Negate a simple sentence that has no auxiliary verb by inserting
    do/does/did + not before the main verb.

    Tries spaCy first (accurate tense/lemma); falls back to regex heuristics.
    Returns None if the main verb cannot be confidently identified
    (e.g. noun phrases, sentence fragments).
    """
    # ── spaCy path ────────────────────────────────────────────────────────
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        root = next(
            (t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"),
            None,
        )
        if root is None:
            return None   # no main verb → likely a noun phrase, skip
        morph = str(root.morph)
        if "Tense=Past" in morph:
            aux = "did not"
        elif "Person=3" in morph and "Number=Sing" in morph:
            aux = "does not"
        else:
            aux = "do not"
        s, e = root.idx, root.idx + len(root.text)
        return text[:s] + aux + " " + root.lemma_ + text[e:]

    # ── Regex fallback (spaCy unavailable) ───────────────────────────────
    # Scan tokens left-to-right; skip the first token (likely subject start).
    # Recognise: irregular past → did not, -ed → did not, -s/-es → does not.
    # Base-form verbs are too ambiguous to detect without POS tags → skip.
    _SKIP = frozenset({
        "a", "an", "the", "this", "that", "these", "those",
        "i", "he", "she", "it", "we", "they", "you",
        "my", "his", "her", "its", "our", "their", "your",
        "some", "any", "all", "both", "each", "every",
    })
    # 常见以 -s/-es 结尾但本质上是介词/副词/名词的词，避免误判
    _NON_VERB_S = frozenset({
        "towards", "backwards", "forwards", "downwards", "upwards",
        "inwards", "outwards", "afterwards", "onwards", "sideways",
        "across", "unless", "thus", "plus", "minus", "versus",
        "always", "sometimes", "perhaps", "series", "species",
        "news", "yes", "its",
    })
    # 以这些后缀结尾的词几乎不是动词第三人称单数
    _NON_VERB_ENDS = ("wards", "ward", "tion", "sion", "ness",
                      "ment", "less", "ous", "ics", "ics")

    spans = list(re.finditer(r"\S+", text))
    for i, m in enumerate(spans):
        if i == 0:
            continue
        w = re.sub(r"[^\w]", "", m.group()).lower()
        if not w or w in _SKIP:
            continue

        if w in _IRR_PAST_TO_BASE:
            base = _IRR_PAST_TO_BASE[w]
            return text[: m.start()] + "did not " + base + text[m.end():]

        if len(w) > 3 and w.endswith("ed"):
            base = _strip_ed(w)
            return text[: m.start()] + "did not " + base + text[m.end():]

        if (len(w) > 2 and w.endswith("s")
                and w not in _NON_VERB_S
                and not any(w.endswith(e) for e in _NON_VERB_ENDS)
                and not w.endswith("ss")):
            base = _strip_s(w)
            return text[: m.start()] + "does not " + base + text[m.end():]

    return None   # couldn't identify verb confidently


def _normalize_space(s: str) -> str:
    return " ".join(s.strip().split())


def _replace_first(pattern: re.Pattern, text: str, repl: str) -> Optional[str]:
    new_text, count = pattern.subn(repl, text, count=1)
    if count == 0:
        return None
    return new_text


def _parse_number_token(token: str) -> Tuple[Optional[float], bool]:
    """
    Returns (value, is_percent).
    Supports currency prefixes and percent suffix.
    """
    tok = token.strip()
    is_percent = tok.endswith("%")
    # remove currency prefix and commas
    tok2 = re.sub(r"^[$£€]", "", tok)
    tok2 = tok2.replace(",", "")
    tok2 = tok2[:-1] if is_percent else tok2
    try:
        return float(tok2), is_percent
    except ValueError:
        return None, is_percent


def _format_number_like(original_token: str, new_value: float) -> str:
    original = original_token.strip()
    prefix = ""
    if original and original[0] in "$£€":
        prefix = original[0]
    is_percent = original.endswith("%")
    if is_percent:
        return f"{prefix}{new_value:.2f}%"
    # avoid trailing .00 when integer-like
    if abs(new_value - round(new_value)) < 1e-9:
        return f"{prefix}{int(round(new_value))}"
    return f"{prefix}{new_value:.2f}"


def numeric_metric_transform(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Change the first numeric token (number/percent/money) by a small delta.
    """
    m = _NUMBER_PATTERN.search(text)
    if not m:
        return None
    token = m.group(1)
    value, is_percent = _parse_number_token(token)
    if value is None:
        return None

    # Use different deltas for percent vs absolute number.
    delta = 1.0 if not is_percent else 5.0
    new_value = value + delta
    # Prevent accidentally returning same number.
    if abs(new_value - value) < 1e-9:
        new_value = value + (delta if delta != 0 else 1.0)
    repl = _format_number_like(token, new_value)
    return text[: m.start(1)] + repl + text[m.end(1) :]


_PRONOUN_REPLACE = {
    # subject/object/common cases for he/she
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "her",
    # keep plural they/it for safety (or swap them to it as mismatch)
    "they": "it",
    "them": "it",
    "their": "its",
    "it": "they",
    "its": "their",
}


def _replace_pronoun(text: str) -> Optional[str]:
    # Replace only the first pronoun occurrence.
    for pron, repl in _PRONOUN_REPLACE.items():
        # word boundary replacement, case-insensitive.
        pat = re.compile(rf"\b{re.escape(pron)}\b", flags=re.IGNORECASE)
        m = pat.search(text)
        if not m:
            continue
        found = m.group(0)
        # Preserve capitalization (very simple heuristic).
        if found and found[0].isupper():
            repl2 = repl.capitalize()
        else:
            repl2 = repl
        new_text = pat.sub(repl2, text, count=1)
        if new_text != text:
            return new_text
    return None


def entity_pronoun_substitution(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Substitute pronouns/entities to introduce factual mismatch.
    Rule-based: pronoun substitution via he/she swap; entity substitution is conservative fallback.
    """
    # Prefer pronoun substitution.
    out = _replace_pronoun(text)
    if out is not None:
        return out

    # Fallback: entity substitution by swapping first two identical entity spans if available.
    entities = features.get("entities") or []
    if not entities:
        return None
    # Replace the first entity with a generic "another <entity>" pattern.
    # This preserves grammatical form while changing the referent.
    ent = entities[0]
    if not ent:
        return None
    pat = re.compile(rf"\b{re.escape(ent)}\b", flags=re.IGNORECASE)
    if not pat.search(text):
        return None
    return pat.sub(f"another {ent}", text, count=1)


_SCOPE_DEGREE_REPLACE = {
    "all": "some",
    "some": "most",
    "most": "some",
    "many": "few",
    "few": "many",
    "always": "never",
    "never": "always",
    "must": "may",
    "may": "must",
    "might": "must",
    "possibly": "never",
    "maybe": "never",
}


def scope_degree_scaling(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Scale quantifiers and modal/degree words by replacement.
    """
    lower = text.lower()
    for src, dst in _SCOPE_DEGREE_REPLACE.items():
        if re.search(rf"\b{re.escape(src)}\b", lower):
            pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
            m = pat.search(text)
            if not m:
                continue
            found = m.group(0)
            dst2 = dst.capitalize() if found and found[0].isupper() else dst
            new_text = pat.sub(dst2, text, count=1)
            if new_text != text:
                return new_text
    return None


_AUX_VERBS = r"(is|are|was|were|has|have|had|can|could|should|would|will|may|might|do|does|did|must)"
_NEG_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "without"}


def _has_negation(text: str) -> bool:
    lt = text.lower()
    if "n't" in lt:
        return True
    return any(re.search(rf"\b{re.escape(w)}\b", lt) for w in _NEG_WORDS)


def direct_negation_attack(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Flip the truth value of a sentence by manipulating its negation:
    - Positive sentence  → insert 'not' after first auxiliary verb
    - Negated sentence   → remove the negation word to produce the positive claim
      e.g. "There is usually no shooting" → "There is usually shooting"
    """
    if _has_negation(text):
        # ── 删除否定词，将负句变正句 ──────────────────────────────────────
        # 只处理删除后句子仍然通顺的类型：
        #   "no"    → "no shooting" → "shooting"   ✓
        #   "never" → "never goes"  → "goes"        ✓
        #   "not"   → "is not"      → "is"           ✓
        #   缩写    → "doesn't"     → "does"         ✓
        # 不处理 none/nothing/nobody/without（删除后残句语法破坏严重）
        for w in ["no", "never"]:
            pat = re.compile(rf"\b{re.escape(w)}\b\s*", flags=re.IGNORECASE)
            m = pat.search(text)
            if m:
                out = text[: m.start()] + text[m.end():]
                out = re.sub(r"\s{2,}", " ", out).strip()
                if out and out != text:
                    return out
        # 独立的 "not"（含前置空格一起删，避免双空格）
        out = re.sub(r"\s+not\b", "", text, count=1, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip()
        if out and out != text:
            return out
        # 缩写：doesn't → does, isn't → is, can't → can
        out = re.sub(r"\b(\w+?)n['']?t\b", r"\1", text, count=1, flags=re.IGNORECASE)
        if out != text:
            return out
        return None

    # ── 正句：在第一个助动词后插入 not ──────────────────────────────────
    m = re.search(rf"\b{_AUX_VERBS}\b", text, flags=re.IGNORECASE)
    if m:
        aux = m.group(0)
        return text[: m.start()] + f"{aux} not" + text[m.end():]

    # Fallback: 尝试插入 do/does/did not（比 "Not X..." 语法更正确）
    # 无法识别主动词时返回 None，宁可跳过也不产生语法错误输出
    return _negate_without_aux(text)


def double_negation_attack(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Flip polarity by removing an existing negation token (approximation of double-negation effects).
    """
    if not _has_negation(text):
        return None

    # Handle contractions: "doesn't" -> "does"
    out = re.sub(r"\b(\w+?)n['’]?t\b", r"\1", text, count=1, flags=re.IGNORECASE)
    if out != text:
        return out

    # Remove first standalone 'not'.
    out2 = re.sub(r"\bnot\b", "", text, count=1, flags=re.IGNORECASE)
    if out2 != text:
        return re.sub(r"\s{2,}", " ", out2).strip()

    # Remove other neg words
    for w in ["no", "never", "none", "nothing", "nobody", "without"]:
        pat = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
        if pat.search(text):
            out3 = pat.sub("", text, count=1)
            return re.sub(r"\s{2,}", " ", out3).strip()
    return None


_LOGIC_REPLACE = {
    "because": "although",
    "although": "because",
    "if": "unless",
    "unless": "if",
    "therefore": "however",
    "however": "therefore",
    "so": "because",
    "since": "while",
    "when": "while",
    "while": "when",
}


def logical_operator_rewrite(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Rewrite common logical connectors to invert/perturb relation.
    """
    lt = text.lower()
    for src, dst in _LOGIC_REPLACE.items():
        if re.search(rf"\b{re.escape(src)}\b", lt):
            pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
            m = pat.search(text)
            if not m:
                continue
            found = m.group(0)
            dst2 = dst.capitalize() if found and found[0].isupper() else dst
            new_text = pat.sub(dst2, text, count=1)
            if new_text != text:
                return new_text
    return None


def _swap_spans(text: str, a: str, b: str) -> Optional[str]:
    if not a or not b or a.strip() == b.strip():
        return None
    if a.lower() not in text.lower() or b.lower() not in text.lower():
        return None

    placeholder = "__SWAP_PLACEHOLDER__"
    # replace first a -> placeholder
    pat_a = re.compile(rf"\b{re.escape(a)}\b", flags=re.IGNORECASE)
    pat_b = re.compile(rf"\b{re.escape(b)}\b", flags=re.IGNORECASE)
    if not pat_a.search(text) or not pat_b.search(text):
        return None
    step1 = pat_a.sub(placeholder, text, count=1)
    step2 = pat_b.sub(a, step1, count=1)
    step3 = step2.replace(placeholder, b, 1)
    return step3


def role_swap(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Swap subject and object candidates if they appear as substrings.
    """
    subjects = features.get("subject_candidates") or []
    objects = features.get("object_candidates") or []
    if not subjects or not objects:
        return None

    a = subjects[0]
    b = objects[0]
    out = _swap_spans(text, a, b)
    return out


_TEMPORAL_REPLACE = {
    "before": "after",
    "after": "before",
    "then": "after",
    "previously": "later",
    "later": "previously",
    "finally": "first",
    "first": "finally",
}


def temporal_causal_inversion(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Swap basic sequence markers (before/after/then/first/finally...).
    """
    lt = text.lower()
    for src, dst in _TEMPORAL_REPLACE.items():
        if re.search(rf"\b{re.escape(src)}\b", lt):
            pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
            m = pat.search(text)
            if not m:
                continue
            found = m.group(0)
            dst2 = dst.capitalize() if found and found[0].isupper() else dst
            new_text = pat.sub(dst2, text, count=1)
            if new_text != text:
                return new_text
    return None


_CONCEPT_SHIFT_MAP = {
    "apple": "fruit",
    "apples": "fruits",
    "dog": "animal",
    "dogs": "animals",
    "car": "vehicle",
    "cars": "vehicles",
    "city": "place",
    "cities": "places",
    "doctor": "person",
    "doctors": "people",
    "movie": "film",
    "movies": "films",
}


def concept_hierarchy_shift(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Replace some common concepts with a higher-level category (small built-in dictionary).
    """
    lt = text.lower()
    for src, dst in _CONCEPT_SHIFT_MAP.items():
        if re.search(rf"\b{re.escape(src)}\b", lt):
            pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
            m = pat.search(text)
            if not m:
                continue
            found = m.group(0)
            dst2 = dst.capitalize() if found and found[0].isupper() else dst
            new_text = pat.sub(dst2, text, count=1)
            if new_text != text:
                return new_text
    return None


def premise_disruption(formatted_item: Dict[str, Any], text: str, features: Dict[str, Any], llm_engine: Optional[LocalLLMEngine] = None) -> Optional[str]:
    """
    Break premise by removing or corrupting the reason/condition clause.
    """
    # Remove the first "because ...," clause.
    if re.search(r"\bbecause\b", text, flags=re.IGNORECASE):
        out = re.sub(r"\bbecause\b[^,.]*[,]?", "", text, count=1, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip()
        if out and out != text:
            return out

    # If we have an "if" clause, flip it to "even if" / "even though" style.
    if re.search(r"\bif\b", text, flags=re.IGNORECASE):
        out = re.sub(r"\bif\b", "even if", text, count=1, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip()
        if out != text:
            return out

    # Fallback: add a weak modifier to premise.
    stripped = text.strip()
    if not stripped:
        return None
    out = "In theory, " + stripped[0].upper() + stripped[1:]
    return out if out != text else None


# Registry for dispatcher.
METHOD_REGISTRY: Dict[str, Callable[..., Optional[str]]] = {
    "numeric_metric_transform": numeric_metric_transform,
    "entity_pronoun_substitution": entity_pronoun_substitution,
    "scope_degree_scaling": scope_degree_scaling,
    "direct_negation_attack": direct_negation_attack,
    "double_negation_attack": double_negation_attack,
    "logical_operator_rewrite": logical_operator_rewrite,
    "role_swap": role_swap,
    "temporal_causal_inversion": temporal_causal_inversion,
    "concept_hierarchy_shift": concept_hierarchy_shift,
    "premise_disruption": premise_disruption,
}


def apply_method(
    method_name: str,
    formatted_item: Dict[str, Any],
    text: str,
    features: Dict[str, Any],
    llm_engine: Optional[LocalLLMEngine] = None,
) -> Optional[str]:
    fn = METHOD_REGISTRY.get(method_name)
    if fn is None:
        raise ValueError(f"Unknown method_name: {method_name}")
    return fn(formatted_item=formatted_item, text=text, features=features, llm_engine=llm_engine)
