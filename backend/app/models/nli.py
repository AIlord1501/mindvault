"""Local NLI adapter with strict offline guarantees and deterministic heuristic fallback."""

from __future__ import annotations

import importlib
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "use",
        "uses",
        "using",
        "was",
        "with",
    }
)
_NEGATIONS = frozenset({"not", "never", "no", "none", "without", "cannot", "cant", "never"})
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("offline", "online"),
    ("local", "remote"),
    ("local", "cloud"),
    ("enabled", "disabled"),
    ("enable", "disable"),
    ("encrypted", "unencrypted"),
    ("allow", "forbid"),
    ("allows", "forbids"),
    ("true", "false"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("supports", "rejects"),
    ("required", "optional"),
)


class NLIUnavailableError(RuntimeError):
    """Raised when a local NLI model is unavailable offline."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 6)))


@dataclass(frozen=True)
class NLIScores:
    """Three-way NLI relationship probability scores."""

    contradiction: float
    entailment: float
    neutral: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "contradiction", _clamp01(self.contradiction))
        object.__setattr__(self, "entailment", _clamp01(self.entailment))
        object.__setattr__(self, "neutral", _clamp01(self.neutral))

    def dominant_label(
        self,
        *,
        contradiction_threshold: float = 0.70,
        entailment_threshold: float = 0.70,
    ) -> str:
        """Classify relationship against configured contradiction & entailment thresholds."""
        if self.contradiction >= contradiction_threshold and self.contradiction >= self.entailment:
            return "contradiction"
        if self.entailment >= entailment_threshold and self.entailment > self.contradiction:
            return "entailment"
        return "neutral"


class BaseNLIAdapter(ABC):
    """Contract for local Natural Language Inference models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return configured model identifier."""

    @abstractmethod
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Return (available, reason_if_unavailable) without network access."""

    @abstractmethod
    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Predict NLI scores for (premise, hypothesis) text pairs."""


class LocalSequenceClassificationNLIAdapter(BaseNLIAdapter):
    """Offline HuggingFace sequence-classification NLI adapter (e.g. DeBERTa-v3 NLI)."""

    def __init__(
        self,
        *,
        model_path: str = "",
        models_dir: Path = Path("data/models"),
        offline_only: bool = True,
        loader: Optional[Callable[[str], tuple[Any, Any]]] = None,
    ) -> None:
        self._raw_model = model_path.strip()
        self._models_dir = models_dir
        self._offline_only = offline_only
        self._loader = loader
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._resolved_target: Optional[str] = None
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        return self._resolved_target or self._raw_model

    def _resolve_target(self) -> str:
        if not self._raw_model:
            raise NLIUnavailableError(
                "No local NLI model configured. Set MINDVAULT_NLI_MODEL_PATH."
            )

        direct = Path(self._raw_model).expanduser()
        if direct.exists():
            return str(direct)

        in_models_dir = (self._models_dir / self._raw_model).expanduser()
        if in_models_dir.exists():
            return str(in_models_dir)

        looks_like_path = (
            self._raw_model.startswith((".", "/", "\\"))
            or "\\" in self._raw_model
            or ":" in self._raw_model
        )
        if looks_like_path or (self._offline_only and self._loader is None):
            raise NLIUnavailableError(
                f"Configured local NLI model path does not exist offline: {self._raw_model}"
            )
        return self._raw_model

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            if self._load_error is not None:
                raise NLIUnavailableError(self._load_error)

            try:
                target = self._resolve_target()
                if self._offline_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"

                if self._loader is not None:
                    tokenizer, model = self._loader(target)
                else:
                    transformers = importlib.import_module("transformers")
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        target,
                        local_files_only=True,
                    )
                    model = transformers.AutoModelForSequenceClassification.from_pretrained(
                        target,
                        local_files_only=True,
                    )
                    if hasattr(model, "eval"):
                        model.eval()

                self._tokenizer = tokenizer
                self._model = model
                self._resolved_target = target
                return tokenizer, model
            except NLIUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except Exception as exc:
                msg = f"Failed to load local NLI model '{self._raw_model}' offline: {exc}"
                self._load_error = msg
                raise NLIUnavailableError(msg) from exc

    def is_available(self) -> tuple[bool, Optional[str]]:
        if not self._raw_model:
            return False, "No local NLI model configured. Set MINDVAULT_NLI_MODEL_PATH."
        if self._model is not None:
            return True, None
        if self._load_error is not None:
            return False, self._load_error
        try:
            self._ensure_loaded()
            return True, None
        except NLIUnavailableError as exc:
            return False, str(exc)

    @staticmethod
    def _extract_label_indices(id2label: dict[Any, Any]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for raw_idx, raw_label in id2label.items():
            idx = int(raw_idx)
            label = str(raw_label).strip().lower()
            if "contradict" in label:
                mapping["contradiction"] = idx
            elif "entail" in label:
                mapping["entailment"] = idx
            elif "neutral" in label:
                mapping["neutral"] = idx
        if len(mapping) < 3:
            return {"contradiction": 0, "neutral": 1, "entailment": 2}
        return mapping

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        tokenizer, model = self._ensure_loaded()
        torch = importlib.import_module("torch")

        premises = [p for p, _h in pairs]
        hypotheses = [h for _p, h in pairs]
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).tolist()

        id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
        idx_map = self._extract_label_indices(id2label)

        results: list[NLIScores] = []
        for row in probs:
            results.append(
                NLIScores(
                    contradiction=float(row[idx_map["contradiction"]]),
                    entailment=float(row[idx_map["entailment"]]),
                    neutral=float(row[idx_map["neutral"]]),
                )
            )
        return results


class HeuristicLexicalNLIAdapter(BaseNLIAdapter):
    """Deterministic offline fallback when no neural NLI model is configured."""

    @property
    def model_name(self) -> str:
        return "heuristic-lexical-nli"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _score_pair(premise: str, hypothesis: str) -> NLIScores:
        p_tokens = _TOKEN_RE.findall(premise.lower())
        h_tokens = _TOKEN_RE.findall(hypothesis.lower())

        p_content = {t for t in p_tokens if t not in _STOPWORDS and t not in _NEGATIONS}
        h_content = {t for t in h_tokens if t not in _STOPWORDS and t not in _NEGATIONS}
        shared = p_content & h_content
        union_size = len(p_content | h_content) or 1
        overlap_ratio = len(shared) / union_size

        p_neg = bool(set(p_tokens) & _NEGATIONS)
        h_neg = bool(set(h_tokens) & _NEGATIONS)
        neg_flip = p_neg != h_neg

        p_set = set(p_tokens)
        h_set = set(h_tokens)
        antonym_hit = any(
            (a in p_set and b in h_set) or (b in p_set and a in h_set)
            for a, b in _ANTONYM_PAIRS
        )

        p_nums = set(_NUMBER_RE.findall(premise))
        h_nums = set(_NUMBER_RE.findall(hypothesis))
        numeric_conflict = bool(p_nums and h_nums and p_nums != h_nums)

        has_topic_overlap = len(shared) >= 2 or overlap_ratio >= 0.25
        if has_topic_overlap and (neg_flip or antonym_hit or numeric_conflict):
            return NLIScores(contradiction=0.82, entailment=0.06, neutral=0.12)

        min_len = min(len(p_content), len(h_content)) or 1
        inclusion = len(shared) / min_len
        if inclusion >= 0.65 and not neg_flip and not antonym_hit and not numeric_conflict:
            return NLIScores(contradiction=0.05, entailment=0.80, neutral=0.15)

        return NLIScores(contradiction=0.08, entailment=0.12, neutral=0.80)

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        return [self._score_pair(p, h) for p, h in pairs]
