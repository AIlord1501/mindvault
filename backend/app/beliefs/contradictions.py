"""Semantic-filtered NLI contradiction & entailment detection across beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.beliefs.extractor import cosine_similarity, lexical_jaccard
from app.beliefs.repository import BeliefRepository
from app.models.nli import BaseNLIAdapter, HeuristicLexicalNLIAdapter


@dataclass(frozen=True)
class DetectionSummary:
    """Result returned by ContradictionDetector after evaluating new beliefs."""

    nli_mode: str
    nli_model: str
    candidates_evaluated: int
    relations_recorded: list[dict[str, Any]]
    contradictions_created: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


class ContradictionDetector:
    """Compares new beliefs against semantically filtered active beliefs using local NLI."""

    def __init__(
        self,
        *,
        repo: BeliefRepository,
        nli_adapter: BaseNLIAdapter,
        fallback_nli: Optional[BaseNLIAdapter] = None,
        candidate_top_k: int = 10,
        candidate_min_similarity: float = 0.35,
        contradiction_threshold: float = 0.70,
        entailment_threshold: float = 0.70,
    ) -> None:
        self._repo = repo
        self._nli = nli_adapter
        self._fallback_nli = fallback_nli or HeuristicLexicalNLIAdapter()
        self._candidate_top_k = max(1, int(candidate_top_k))
        self._candidate_min_sim = float(candidate_min_similarity)
        self._contradiction_threshold = float(contradiction_threshold)
        self._entailment_threshold = float(entailment_threshold)

    def _select_candidates(
        self,
        *,
        new_belief: dict[str, Any],
        new_embedding: Optional[list[float]],
        active_pool: list[dict[str, Any]],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[dict[str, Any], float]]:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        new_id = str(new_belief["id"])
        new_text = str(new_belief["belief_text"])

        for candidate in active_pool:
            cand_id = str(candidate["id"])
            if cand_id == new_id:
                continue

            cand_vec = candidate.get("embedding")
            if new_embedding and isinstance(cand_vec, list) and len(cand_vec) == len(new_embedding):
                sim = cosine_similarity(new_embedding, cand_vec)
                threshold = min_similarity
            else:
                sim = lexical_jaccard(new_text, str(candidate["belief_text"]))
                # Lexical Jaccard on short <=20-word sentences is sparser than cosine similarity.
                threshold = min(min_similarity, 0.20)

            if sim >= threshold:
                scored.append((round(float(sim), 6), cand_id, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(cand, sim) for sim, _cid, cand in scored[:top_k]]

    def evaluate_new_beliefs(
        self,
        *,
        new_beliefs: list[tuple[dict[str, Any], Optional[list[float]]]],
        active_pool: list[dict[str, Any]],
        candidate_top_k: Optional[int] = None,
        candidate_min_similarity: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
        entailment_threshold: Optional[float] = None,
    ) -> DetectionSummary:
        """Compare each newly extracted belief against top-K semantically similar active beliefs."""
        if not new_beliefs or not active_pool:
            available, _ = self._nli.is_available()
            return DetectionSummary(
                nli_mode="local_nli_model" if available else "heuristic_fallback",
                nli_model=self._nli.model_name if available else self._fallback_nli.model_name,
                candidates_evaluated=0,
                relations_recorded=[],
                contradictions_created=[],
                warnings=[],
            )

        top_k = self._candidate_top_k if candidate_top_k is None else max(1, int(candidate_top_k))
        min_sim = (
            self._candidate_min_sim
            if candidate_min_similarity is None
            else float(candidate_min_similarity)
        )
        contra_thresh = (
            self._contradiction_threshold
            if contradiction_threshold is None
            else float(contradiction_threshold)
        )
        entail_thresh = (
            self._entailment_threshold
            if entailment_threshold is None
            else float(entailment_threshold)
        )

        warnings: list[str] = []
        nli_available, nli_reason = self._nli.is_available()
        if nli_available:
            active_nli = self._nli
            nli_mode = "local_nli_model"
        else:
            active_nli = self._fallback_nli
            nli_mode = "heuristic_fallback"
            warnings.append(
                "Local neural NLI model unavailable; used heuristic lexical NLI fallback: "
                f"{nli_reason or 'model not configured.'}"
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        total_evaluated = 0
        relations_recorded: list[dict[str, Any]] = []
        contradictions_created: list[dict[str, Any]] = []

        for new_belief, new_vec in new_beliefs:
            candidates = self._select_candidates(
                new_belief=new_belief,
                new_embedding=new_vec,
                active_pool=active_pool,
                top_k=top_k,
                min_similarity=min_sim,
            )
            if not candidates:
                continue

            pairs = [
                (str(cand["belief_text"]), str(new_belief["belief_text"]))
                for cand, _sim in candidates
            ]
            predictions = active_nli.predict_pairs(pairs)
            total_evaluated += len(pairs)

            for (cand, sim_score), scores in zip(candidates, predictions):
                label = scores.dominant_label(
                    contradiction_threshold=contra_thresh,
                    entailment_threshold=entail_thresh,
                )
                relation = self._repo.save_relation(
                    source_belief_id=str(cand["id"]),
                    target_belief_id=str(new_belief["id"]),
                    label=label,
                    contradiction_score=scores.contradiction,
                    entailment_score=scores.entailment,
                    neutral_score=scores.neutral,
                    similarity_score=sim_score,
                    created_at=now_iso,
                )
                relations_recorded.append(relation)

                if label == "contradiction":
                    contradiction = self._repo.create_contradiction(
                        belief_x_id=str(cand["id"]),
                        belief_y_id=str(new_belief["id"]),
                        score=scores.contradiction,
                        entailment_score=scores.entailment,
                        neutral_score=scores.neutral,
                        similarity_score=sim_score,
                        created_at=now_iso,
                    )
                    contradictions_created.append(contradiction)

        return DetectionSummary(
            nli_mode=nli_mode,
            nli_model=active_nli.model_name,
            candidates_evaluated=total_evaluated,
            relations_recorded=relations_recorded,
            contradictions_created=contradictions_created,
            warnings=warnings,
        )
