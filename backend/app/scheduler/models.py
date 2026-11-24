"""Extensible retention models and offline flashcard generation for spaced repetition."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from app.core.config import Settings

ReviewOutcome = Literal["forgot", "hard", "good", "easy"]

OUTCOME_TO_RATING: dict[str, int] = {
    "forgot": 1,
    "hard": 2,
    "good": 3,
    "easy": 4,
}
RATING_TO_OUTCOME: dict[int, str] = {v: k for k, v in OUTCOME_TO_RATING.items()}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "among",
        "and",
        "another",
        "any",
        "are",
        "around",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "does",
        "during",
        "each",
        "either",
        "enough",
        "every",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "here",
        "how",
        "however",
        "into",
        "its",
        "just",
        "more",
        "most",
        "much",
        "must",
        "never",
        "not",
        "only",
        "other",
        "our",
        "out",
        "over",
        "same",
        "should",
        "since",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "using",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "without",
        "would",
        "your",
    }
)


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def format_iso_datetime(dt: datetime) -> str:
    """Serialize a datetime in normalized UTC ISO-8601 format."""
    if dt.tzinfo is None:
        normalized = dt.replace(tzinfo=timezone.utc)
    else:
        normalized = dt.astimezone(timezone.utc)
    return normalized.isoformat()


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a UTC-aware datetime."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("Timestamp string must not be empty.")
    if cleaned.endswith("Z") or cleaned.endswith("z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_outcome_and_rating(
    outcome: Optional[str] = None,
    rating: Optional[int] = None,
) -> tuple[str, int]:
    """Validate and resolve a review outcome string and 1-4 rating pair."""
    normalized_outcome: Optional[str] = None
    if outcome is not None:
        candidate = outcome.strip().lower()
        if candidate not in OUTCOME_TO_RATING:
            allowed = ", ".join(OUTCOME_TO_RATING.keys())
            raise ValueError(f"Invalid review outcome '{outcome}'. Expected one of: {allowed}.")
        normalized_outcome = candidate

    normalized_rating: Optional[int] = None
    if rating is not None:
        if rating not in RATING_TO_OUTCOME:
            raise ValueError(f"Invalid review rating '{rating}'. Expected integer in 1..4.")
        normalized_rating = rating

    if normalized_outcome is None and normalized_rating is None:
        raise ValueError(
            "Either 'outcome' ('forgot'|'hard'|'good'|'easy') or 'rating' (1..4) is required."
        )

    if normalized_outcome is not None and normalized_rating is not None:
        expected_rating = OUTCOME_TO_RATING[normalized_outcome]
        if normalized_rating != expected_rating:
            raise ValueError(
                f"Conflicting outcome '{normalized_outcome}' (rating {expected_rating}) "
                f"and rating {normalized_rating}."
            )
        return normalized_outcome, normalized_rating

    if normalized_outcome is not None:
        return normalized_outcome, OUTCOME_TO_RATING[normalized_outcome]

    assert normalized_rating is not None
    return RATING_TO_OUTCOME[normalized_rating], normalized_rating


class BaseRetentionModel(ABC):
    """Strategy interface for pluggable memory retention and spacing models."""

    model_name: str = "base"

    @abstractmethod
    def compute_elapsed_days(self, reference_iso: str, now: datetime) -> float:
        """Compute elapsed time t in days since the reference timestamp."""

    @abstractmethod
    def compute_retention(self, elapsed_days: float, stability_days: float) -> float:
        """Estimate retention R in [0.0, 1.0] given elapsed time t and stability S."""

    @abstractmethod
    def compute_next_interval_days(self, stability_days: float) -> float:
        """Calculate days until retention reaches the review eligibility threshold."""

    @abstractmethod
    def update_stability(
        self,
        current_stability: float,
        outcome: str,
        retention_now: float,
    ) -> float:
        """Update memory stability S after recording a recall review outcome."""

    @abstractmethod
    def is_eligible_for_review(
        self,
        retention: float,
        next_review_at: Optional[str],
        now: datetime,
    ) -> bool:
        """Return True if a chunk is eligible/due for review at timestamp now."""


class ExponentialDecayRetentionModel(BaseRetentionModel):
    """Exponential memory decay model: R = exp(-t / S)."""

    model_name: str = "exponential_decay"

    def __init__(
        self,
        *,
        initial_stability_days: float = 1.0,
        min_stability_days: float = 0.25,
        max_stability_days: float = 3650.0,
        retention_threshold: float = 0.75,
        factor_forgot: float = 0.50,
        factor_hard: float = 1.20,
        factor_good: float = 2.20,
        factor_easy: float = 3.20,
    ) -> None:
        if min_stability_days <= 0.0:
            raise ValueError("min_stability_days must be positive.")
        if initial_stability_days <= 0.0:
            raise ValueError("initial_stability_days must be positive.")
        if min_stability_days > initial_stability_days:
            raise ValueError("min_stability_days must be <= initial_stability_days.")
        if max_stability_days < initial_stability_days:
            raise ValueError("max_stability_days must be >= initial_stability_days.")
        if not (0.0 < retention_threshold < 1.0):
            raise ValueError("retention_threshold must be strictly between 0.0 and 1.0.")
        if not (0.0 < factor_forgot <= 1.0):
            raise ValueError("factor_forgot must be in (0.0, 1.0].")
        if factor_hard < 1.0:
            raise ValueError("factor_hard must be >= 1.0.")
        if factor_good < factor_hard:
            raise ValueError("factor_good must be >= factor_hard.")
        if factor_easy < factor_good:
            raise ValueError("factor_easy must be >= factor_good.")

        self.initial_stability_days = float(initial_stability_days)
        self.min_stability_days = float(min_stability_days)
        self.max_stability_days = float(max_stability_days)
        self.retention_threshold = float(retention_threshold)
        self.factor_forgot = float(factor_forgot)
        self.factor_hard = float(factor_hard)
        self.factor_good = float(factor_good)
        self.factor_easy = float(factor_easy)

    @classmethod
    def from_settings(cls, settings: Settings) -> ExponentialDecayRetentionModel:
        """Instantiate the exponential decay model from application Settings."""
        return cls(
            initial_stability_days=settings.retention_initial_stability_days,
            min_stability_days=settings.retention_min_stability_days,
            max_stability_days=settings.retention_max_stability_days,
            retention_threshold=settings.retention_threshold,
            factor_forgot=settings.retention_factor_forgot,
            factor_hard=settings.retention_factor_hard,
            factor_good=settings.retention_factor_good,
            factor_easy=settings.retention_factor_easy,
        )

    def compute_elapsed_days(self, reference_iso: str, now: datetime) -> float:
        """Compute non-negative elapsed days t = (now - ref) / 86400."""
        ref_dt = parse_iso_datetime(reference_iso)
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        delta_seconds = (now_utc.astimezone(timezone.utc) - ref_dt).total_seconds()
        return max(0.0, delta_seconds / 86400.0)

    def compute_retention(self, elapsed_days: float, stability_days: float) -> float:
        """Evaluate R = exp(-t / S) clamped to [0.0, 1.0]."""
        if stability_days <= 0.0:
            raise ValueError(f"stability_days must be positive, got {stability_days}.")
        t = max(0.0, float(elapsed_days))
        raw = math.exp(-t / float(stability_days))
        return max(0.0, min(1.0, raw))

    def compute_next_interval_days(self, stability_days: float) -> float:
        """Solve exp(-dt / S) = R_threshold => dt = -S * ln(R_threshold)."""
        if stability_days <= 0.0:
            raise ValueError(f"stability_days must be positive, got {stability_days}.")
        return -float(stability_days) * math.log(self.retention_threshold)

    def update_stability(
        self,
        current_stability: float,
        outcome: str,
        retention_now: float,
    ) -> float:
        """Update memory stability S based on recall outcome and current retention."""
        canonical_outcome, _ = resolve_outcome_and_rating(outcome=outcome)
        s = max(self.min_stability_days, float(current_stability))
        r = max(0.0, min(1.0, float(retention_now)))

        if canonical_outcome == "forgot":
            updated = max(self.min_stability_days, s * self.factor_forgot)
        elif canonical_outcome == "hard":
            updated = min(
                self.max_stability_days,
                max(self.min_stability_days, s * self.factor_hard),
            )
        elif canonical_outcome == "good":
            multiplier = 1.0 + (self.factor_good - 1.0) * (1.0 + 0.25 * (1.0 - r))
            updated = min(
                self.max_stability_days,
                max(self.min_stability_days, s * multiplier),
            )
        else:  # easy
            multiplier = 1.0 + (self.factor_easy - 1.0) * (1.0 + 0.35 * (1.0 - r))
            updated = min(
                self.max_stability_days,
                max(self.min_stability_days, s * multiplier),
            )

        return round(updated, 6)

    def is_eligible_for_review(
        self,
        retention: float,
        next_review_at: Optional[str],
        now: datetime,
    ) -> bool:
        """Return True when retention <= threshold or scheduled review time has arrived."""
        if float(retention) <= self.retention_threshold + 1e-9:
            return True
        if next_review_at:
            now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
            if parse_iso_datetime(next_review_at) <= now_utc.astimezone(timezone.utc):
                return True
        return False


def _extract_anchor_keywords(text: str, *, max_keywords: int = 3) -> list[str]:
    """Select representative non-stopword tokens without reproducing sentences."""
    words = _WORD_RE.findall(text or "")
    if len(words) <= 3:
        return []

    seen: set[str] = set()
    candidates: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in _STOPWORDS or len(lower) < 4 or lower in seen:
            continue
        seen.add(lower)
        candidates.append(word)

    if not candidates:
        return []
    # Pick non-consecutive terms sorted slightly toward informative longer tokens
    candidates.sort(key=lambda w: len(w), reverse=True)
    return candidates[:max_keywords]


def _build_cloze_hint_from_belief(belief_text: str) -> str:
    """Create a masked recall cue from a belief without leaking the full claim."""
    tokens = belief_text.strip().split()
    if len(tokens) <= 3:
        return ""
    keep_head = 2 if len(tokens) >= 5 else 1
    masked_tail = " ".join("____" for _ in tokens[keep_head : min(len(tokens), keep_head + 4)])
    return f"{' '.join(tokens[:keep_head])} {masked_tail}"


def build_flashcard_for_chunk(
    chunk: dict[str, Any],
    beliefs: list[dict[str, Any]],
    *,
    include_answer: bool = False,
) -> dict[str, Any]:
    """Build a self-recall flashcard for a chunk, optionally revealing its answer."""
    filename = str(chunk.get("filename") or "document")
    chunk_index = int(chunk.get("chunk_index", 0))
    epoch = int(chunk.get("epoch", 0))
    content = str(chunk.get("content") or "").strip()
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    section_heading = str(metadata.get("section_heading") or "").strip()

    content_norm = " ".join(content.lower().split())
    belief_texts = [str(b["belief_text"]).strip() for b in beliefs if b.get("belief_text")]

    prompt: str
    hint: str
    if (
        section_heading
        and section_heading.lower() not in content_norm
        and len(section_heading) <= 80
    ):
        prompt = (
            f"Recall the key details from '{filename}' (chunk #{chunk_index}) "
            f"under section '{section_heading}'."
        )
        hint = f"Section: {section_heading}"
    elif section_heading and section_heading.lower() != content_norm:
        keywords = _extract_anchor_keywords(content, max_keywords=2)
        if keywords:
            prompt = (
                f"Recall what '{filename}' (chunk #{chunk_index}) explains "
                f"in section '{section_heading}' regarding: {', '.join(keywords)}."
            )
        else:
            prompt = (
                f"Recall the key concepts from '{filename}' (chunk #{chunk_index}) "
                f"in section '{section_heading}'."
            )
        hint = f"Section: {section_heading}"
    else:
        keywords = _extract_anchor_keywords(content, max_keywords=3)
        cloze = _build_cloze_hint_from_belief(belief_texts[0]) if belief_texts else ""
        if cloze and cloze.lower() not in content_norm:
            prompt = (
                f"Recall the knowledge recorded in '{filename}' (chunk #{chunk_index}). "
                f"Cue: {cloze}"
            )
            hint = f"Topics: {', '.join(keywords)}" if keywords else f"Source: {filename}"
        elif keywords:
            prompt = (
                f"Recall the facts from '{filename}' (chunk #{chunk_index}) "
                f"concerning topics: {', '.join(keywords)}."
            )
            hint = f"Topics: {', '.join(keywords)}"
        else:
            prompt = (
                f"Recall the note recorded in '{filename}' "
                f"(chunk #{chunk_index}, epoch {epoch})."
            )
            hint = f"Source: {filename} (chunk #{chunk_index})"

    # Safety check: never let prompt reproduce the full chunk content verbatim.
    if content_norm and content_norm in " ".join(prompt.lower().split()):
        prompt = (
            f"Recall the note recorded in '{filename}' "
            f"(chunk #{chunk_index}, epoch {epoch})."
        )

    return {
        "prompt": prompt,
        "hint": hint,
        "answer_revealed": bool(include_answer),
        "answer": content if include_answer else None,
        "key_beliefs": belief_texts if include_answer else None,
    }
