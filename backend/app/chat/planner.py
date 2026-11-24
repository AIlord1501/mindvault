"""Bounded query complexity analyzer and subquestion planner (zero recursive loops)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.models.llm import BaseLLMProvider, CancelCheck, LLMCancelledError

MAX_SUBQUESTIONS_HARD_CAP = 5

_COMPARE_PATTERN = re.compile(
    r"(?:compare|contrast|difference\s+between)\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+?)"
    r"(?:[?.!]|$)",
    re.IGNORECASE,
)
_VS_PATTERN = re.compile(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)", re.IGNORECASE)
_AND_INTERROGATIVE_RE = re.compile(
    r"\s+(?:and|also)\s+(?=(?:what|how|why|when|where|which|who)\b)",
    re.IGNORECASE,
)
_COMPLEXITY_MARKERS = (
    "compare ",
    "contrast ",
    "difference between",
    " versus ",
    " vs ",
    " vs. ",
    "evolution of",
    "pros and cons",
    "trade-offs",
    "tradeoffs",
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


@dataclass(frozen=True)
class QueryPlan:
    """Bounded plan produced before retrieval."""

    original_query: str
    needs_decomposition: bool
    subquestions: list[str]
    strategy: str


def _clean_clause(raw: str) -> str:
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return ""
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!;") + "?"
    return cleaned


def _dedupe_cap(candidates: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        cleaned = _clean_clause(item)
        if len(cleaned) < 4:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= cap:
            break
    return result


def _deterministic_decompose(query: str, cap: int) -> list[str]:
    stripped = query.strip()

    # 1. Explicit multiple questions separated by '?' or ';'
    parts = [p.strip() for p in re.split(r"[?;]+", stripped) if p.strip()]
    if len(parts) > 1:
        expanded: list[str] = []
        for part in parts:
            sub_clauses = [c.strip() for c in _AND_INTERROGATIVE_RE.split(part) if c.strip()]
            expanded.extend(sub_clauses)
        deduped = _dedupe_cap(expanded, cap)
        if len(deduped) > 1:
            return deduped

    # 2. Coordinating interrogative clauses ("What is X and how does Y work?")
    clause_parts = [c.strip() for c in _AND_INTERROGATIVE_RE.split(stripped) if c.strip()]
    if len(clause_parts) > 1:
        deduped = _dedupe_cap(clause_parts, cap)
        if len(deduped) > 1:
            return deduped

    # 3. Comparative / versus prompts ("Compare A and B" or "A vs B")
    comp_match = _COMPARE_PATTERN.search(stripped)
    if comp_match:
        left, right = comp_match.group(1).strip(), comp_match.group(2).strip()
        return _dedupe_cap(
            [
                f"What do the notes state about {left}?",
                f"What do the notes state about {right}?",
                f"What contrasts or relationships exist between {left} and {right}?",
            ],
            cap,
        )

    vs_match = _VS_PATTERN.search(stripped)
    if vs_match:
        left, right = vs_match.group(1).strip(), vs_match.group(2).strip()
        return _dedupe_cap(
            [
                f"What do the notes state about {left}?",
                f"What do the notes state about {right}?",
            ],
            cap,
        )

    return [stripped]


def _is_complex_query(query: str) -> bool:
    lower = query.lower()
    if query.count("?") >= 2 or ";" in query:
        return True
    if any(marker in lower for marker in _COMPLEXITY_MARKERS):
        return True
    if _AND_INTERROGATIVE_RE.search(query):
        return True
    return False


def _single_shot_llm_plan(
    query: str,
    *,
    cap: int,
    llm_provider: BaseLLMProvider,
    cancel_check: Optional[CancelCheck],
) -> list[str]:
    system_prompt = (
        "Break the complex question into at most "
        f"{cap} concise independent subquestions for local document lookup. "
        "Output only one subquestion per line."
    )
    raw = llm_provider.generate(
        system_prompt=system_prompt,
        user_prompt=query,
        max_tokens=128,
        temperature=0.0,
        cancel_check=cancel_check,
    )
    lines = [_BULLET_PREFIX_RE.sub("", line).strip() for line in raw.splitlines()]
    return _dedupe_cap([ln for ln in lines if ln], cap)


def analyze_and_plan_query(
    query: str,
    *,
    max_subquestions: int = 3,
    force_decompose: Optional[bool] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    use_llm_planner: bool = False,
    cancel_check: Optional[CancelCheck] = None,
) -> QueryPlan:
    """Analyze query complexity and produce a strictly bounded retrieval plan."""
    cleaned_query = " ".join((query or "").strip().split())
    if not cleaned_query:
        raise ValueError("Chat query must not be empty.")

    cap = max(1, min(int(max_subquestions), MAX_SUBQUESTIONS_HARD_CAP))

    if force_decompose is False:
        return QueryPlan(
            original_query=cleaned_query,
            needs_decomposition=False,
            subquestions=[],
            strategy="direct",
        )

    complex_detected = _is_complex_query(cleaned_query)
    if not complex_detected and force_decompose is not True:
        return QueryPlan(
            original_query=cleaned_query,
            needs_decomposition=False,
            subquestions=[],
            strategy="direct",
        )

    if (
        use_llm_planner
        and llm_provider is not None
        and llm_provider.is_available()[0]
    ):
        try:
            llm_subqs = _single_shot_llm_plan(
                cleaned_query,
                cap=cap,
                llm_provider=llm_provider,
                cancel_check=cancel_check,
            )
            if len(llm_subqs) > 1 or (force_decompose is True and len(llm_subqs) >= 1):
                return QueryPlan(
                    original_query=cleaned_query,
                    needs_decomposition=True,
                    subquestions=llm_subqs[:cap],
                    strategy="llm_single_shot",
                )
        except LLMCancelledError:
            raise
        except Exception:
            pass

    subquestions = _deterministic_decompose(cleaned_query, cap)
    if len(subquestions) <= 1 and force_decompose is not True:
        return QueryPlan(
            original_query=cleaned_query,
            needs_decomposition=False,
            subquestions=[],
            strategy="direct",
        )

    return QueryPlan(
        original_query=cleaned_query,
        needs_decomposition=True,
        subquestions=subquestions[:cap],
        strategy="deterministic",
    )
