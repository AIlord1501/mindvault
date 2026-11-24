"""Bounded 7-step local retrieval-grounded chat service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from app.chat.planner import analyze_and_plan_query
from app.chat.prompts import (
    MODE_MAX_TOKENS,
    VALID_ANSWER_MODES,
    build_chat_prompts,
    extract_answer_sections,
)
from app.core.config import Settings
from app.db.repositories import DocumentRepository
from app.models.embeddings import EmbeddingUnavailableError
from app.models.llm import (
    BaseLLMProvider,
    CancelCheck,
    LLMCancelledError,
    LLMError,
    LLMUnavailableError,
)
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchFilters, SearchHit


@dataclass(frozen=True)
class CitationRef:
    """Grounded citation reference linking a chat response to a retrieved SQLite chunk."""

    label: str
    citation_id: str
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    source_path: str
    chunk_index: int
    epoch: int
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "epoch": self.epoch,
            "score": self.score,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class ChatResult:
    """Complete chat response returned by POST /api/v1/chat."""

    query_id: str
    query: str
    answer: str
    answer_mode: str
    socratic: bool
    decomposed: bool
    subquestions: list[str]
    citations: list[CitationRef]
    sections: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""


class ChatService:
    """Executes bounded query planning, grounded retrieval, citation storage, and local LLM."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: DocumentRepository,
        retrieval_service: RetrievalService,
        llm_provider: BaseLLMProvider,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._retrieval = retrieval_service
        self._llm = llm_provider

    @staticmethod
    def _check_cancelled(cancel_check: Optional[CancelCheck]) -> None:
        if cancel_check is not None and cancel_check():
            raise LLMCancelledError("Chat workflow cancelled.")

    @staticmethod
    def _make_excerpt(text: str, max_len: int = 220) -> str:
        collapsed = " ".join((text or "").split())
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[: max_len - 3].rstrip() + "..."

    def _retrieve_grounded_chunks(
        self,
        *,
        query: str,
        subquestions: list[str],
        search_mode: str,
        top_k: int,
        filters: Optional[SearchFilters],
        cancel_check: Optional[CancelCheck],
    ) -> tuple[list[SearchHit], list[str]]:
        self._check_cancelled(cancel_check)
        warnings: list[str] = []
        by_chunk_id: dict[str, SearchHit] = {}

        primary = self._retrieval.search(
            query=query,
            mode=search_mode,
            top_k=top_k,
            filters=filters,
        )
        warnings.extend(primary.warnings)
        for hit in primary.hits:
            by_chunk_id[hit.chunk_id] = hit

        sub_top_k = max(2, min(top_k, max(2, top_k // 2)))
        for sq in subquestions:
            self._check_cancelled(cancel_check)
            sub_res = self._retrieval.search(
                query=sq,
                mode=search_mode,
                top_k=sub_top_k,
                filters=filters,
            )
            for w in sub_res.warnings:
                if w not in warnings:
                    warnings.append(w)
            for hit in sub_res.hits:
                existing = by_chunk_id.get(hit.chunk_id)
                if existing is None or hit.score > existing.score:
                    by_chunk_id[hit.chunk_id] = hit

        ranked = sorted(
            by_chunk_id.values(),
            key=lambda h: (-h.score, -h.rrf_score, h.chunk_id),
        )
        limit = min(self._settings.chat_max_context_chunks, max(top_k, len(ranked)))
        return ranked[: self._settings.chat_max_context_chunks][:limit], warnings

    def _run_workflow(
        self,
        *,
        query: str,
        answer_mode: str = "summary",
        socratic: bool = False,
        search_mode: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        decompose: Optional[bool] = None,
        use_llm_planner: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Chat query must not be empty.")
        if answer_mode not in VALID_ANSWER_MODES:
            raise ValueError(f"Unsupported answer_mode '{answer_mode}'.")

        query_id = uuid.uuid4().hex
        resolved_top_k = self._settings.retrieval_top_k if top_k is None else int(top_k)
        resolved_temp = (
            self._settings.llm_temperature if temperature is None else float(temperature)
        )
        mode_cap = MODE_MAX_TOKENS[answer_mode]
        configured_cap = min(self._settings.llm_max_tokens, mode_cap)
        resolved_max_tokens = (
            min(int(max_tokens), self._settings.llm_max_tokens)
            if max_tokens is not None
            else configured_cap
        )

        # 1. Analyze query
        self._check_cancelled(cancel_check)
        yield ("status", {"stage": "analyzing", "query_id": query_id, "message": "Analyzing query"})

        # 2. Bounded query decomposition
        plan = analyze_and_plan_query(
            cleaned_query,
            max_subquestions=self._settings.chat_max_subquestions,
            force_decompose=decompose,
            llm_provider=self._llm,
            use_llm_planner=use_llm_planner,
            cancel_check=cancel_check,
        )
        if plan.needs_decomposition:
            yield (
                "status",
                {
                    "stage": "decomposed",
                    "query_id": query_id,
                    "strategy": plan.strategy,
                    "subquestion_count": len(plan.subquestions),
                },
            )
            for idx, sq in enumerate(plan.subquestions):
                yield ("subquestion", {"index": idx, "question": sq})

        # 3 & 4. Retrieve primary + per-subquestion chunks
        self._check_cancelled(cancel_check)
        yield (
            "status",
            {
                "stage": "retrieving",
                "query_id": query_id,
                "search_mode": search_mode,
            },
        )
        hits, retrieval_warnings = self._retrieve_grounded_chunks(
            query=cleaned_query,
            subquestions=plan.subquestions,
            search_mode=search_mode,
            top_k=resolved_top_k,
            filters=filters,
            cancel_check=cancel_check,
        )

        # 5. Persist and emit grounded citations
        now_iso = datetime.now(timezone.utc).isoformat()
        saved_rows = self._repo.save_citations(
            query_id=query_id,
            chunk_ids=[h.chunk_id for h in hits],
            citation_type="retrieval",
            created_at=now_iso,
        )
        citation_id_by_chunk = {row["chunk_id"]: row["id"] for row in saved_rows}

        citations: list[CitationRef] = []
        evidence_items: list[dict[str, Any]] = []
        for idx, hit in enumerate(hits):
            label = f"C{idx + 1}"
            cit = CitationRef(
                label=label,
                citation_id=citation_id_by_chunk.get(hit.chunk_id, ""),
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                file_type=hit.file_type,
                source_path=hit.source_path,
                chunk_index=hit.chunk_index,
                epoch=hit.epoch,
                score=hit.score,
                excerpt=self._make_excerpt(hit.content),
            )
            citations.append(cit)
            evidence_items.append(
                {
                    "label": label,
                    "filename": hit.filename,
                    "file_type": hit.file_type,
                    "epoch": hit.epoch,
                    "content": hit.content,
                }
            )
            yield ("citation", cit.to_dict())

        # 6. Verify local LLM availability and build hardened prompts
        self._check_cancelled(cancel_check)
        available, reason = self._llm.is_available()
        if not available:
            raise LLMUnavailableError(reason or "Local LLM provider is unavailable.")

        system_prompt, user_prompt = build_chat_prompts(
            query=cleaned_query,
            evidence_items=evidence_items,
            subquestions=plan.subquestions,
            answer_mode=answer_mode,
            socratic=socratic,
        )

        yield (
            "status",
            {
                "stage": "generating",
                "query_id": query_id,
                "provider": self._llm.provider_name,
                "model": self._llm.model_name,
                "answer_mode": answer_mode,
                "socratic": socratic,
            },
        )

        # 7. Stream tokens and assemble final answer + sections
        collected_tokens: list[str] = []
        for token in self._llm.stream_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temp,
            cancel_check=cancel_check,
        ):
            self._check_cancelled(cancel_check)
            collected_tokens.append(token)
            yield ("token", {"text": token})

        full_answer = "".join(collected_tokens)
        sections = extract_answer_sections(full_answer, socratic=socratic)
        for sec in sections:
            yield ("answer_section", sec)

        yield (
            "done",
            {
                "query_id": query_id,
                "status": "completed",
                "query": cleaned_query,
                "answer": full_answer,
                "answer_mode": answer_mode,
                "socratic": socratic,
                "decomposed": plan.needs_decomposition,
                "subquestions": plan.subquestions,
                "citations": [c.to_dict() for c in citations],
                "sections": sections,
                "warnings": retrieval_warnings,
                "provider": self._llm.provider_name,
                "model": self._llm.model_name,
            },
        )

    def chat(
        self,
        *,
        query: str,
        answer_mode: str = "summary",
        socratic: bool = False,
        search_mode: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        decompose: Optional[bool] = None,
        use_llm_planner: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> ChatResult:
        """Execute synchronous local chat and return the complete structured result."""
        done_payload: Optional[dict[str, Any]] = None
        citations: list[CitationRef] = []
        for event_name, payload in self._run_workflow(
            query=query,
            answer_mode=answer_mode,
            socratic=socratic,
            search_mode=search_mode,
            top_k=top_k,
            filters=filters,
            decompose=decompose,
            use_llm_planner=use_llm_planner,
            max_tokens=max_tokens,
            temperature=temperature,
            cancel_check=cancel_check,
        ):
            if event_name == "citation":
                citations.append(CitationRef(**payload))
            elif event_name == "done":
                done_payload = payload

        if done_payload is None:
            raise LLMError("Chat workflow ended without producing a final result.")

        return ChatResult(
            query_id=str(done_payload["query_id"]),
            query=str(done_payload["query"]),
            answer=str(done_payload["answer"]),
            answer_mode=str(done_payload["answer_mode"]),
            socratic=bool(done_payload["socratic"]),
            decomposed=bool(done_payload["decomposed"]),
            subquestions=list(done_payload["subquestions"]),
            citations=citations,
            sections=list(done_payload["sections"]),
            warnings=list(done_payload["warnings"]),
            provider=str(done_payload["provider"]),
            model=str(done_payload["model"]),
        )

    def stream_events(
        self,
        *,
        query: str,
        answer_mode: str = "summary",
        socratic: bool = False,
        search_mode: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        decompose: Optional[bool] = None,
        use_llm_planner: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield SSE events, translating expected errors into structured error + done events."""
        try:
            yield from self._run_workflow(
                query=query,
                answer_mode=answer_mode,
                socratic=socratic,
                search_mode=search_mode,
                top_k=top_k,
                filters=filters,
                decompose=decompose,
                use_llm_planner=use_llm_planner,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_check=cancel_check,
            )
        except LLMCancelledError:
            return
        except (LLMUnavailableError, EmbeddingUnavailableError) as exc:
            yield ("error", {"code": "provider_unavailable", "message": str(exc)})
            yield ("done", {"status": "error", "error": str(exc)})
        except (LLMError, ValueError) as exc:
            yield ("error", {"code": "chat_error", "message": str(exc)})
            yield ("done", {"status": "error", "error": str(exc)})
