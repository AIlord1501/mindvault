"""REST and SSE endpoints for local retrieval-grounded MindVault chat."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.chat.service import ChatService
from app.models.embeddings import EmbeddingUnavailableError
from app.models.llm import (
    LLMCancelledError,
    LLMGenerationError,
    LLMUnavailableError,
)
from app.retrieval.types import SearchFilters

router = APIRouter(tags=["chat"])

_STREAM_END = object()


class ChatRequest(BaseModel):
    """Validated request body for local chat and SSE streaming chat."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4000)
    answer_mode: Literal["brief", "summary", "full"] = "summary"
    socratic: bool = False
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    document_ids: Optional[list[str]] = None
    file_types: Optional[list[str]] = None
    epoch_min: Optional[int] = Field(default=None, ge=0)
    epoch_max: Optional[int] = Field(default=None, ge=0)
    decompose: Optional[bool] = None
    use_llm_planner: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=16, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_constraints(self) -> ChatRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank.")
        if (
            self.epoch_min is not None
            and self.epoch_max is not None
            and self.epoch_min > self.epoch_max
        ):
            raise ValueError("epoch_min must be less than or equal to epoch_max.")
        return self


class CitationResponse(BaseModel):
    """Grounded source chunk citation metadata."""

    model_config = ConfigDict(extra="forbid")

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


class AnswerSectionResponse(BaseModel):
    """Structured section extracted from the generated answer."""

    model_config = ConfigDict(extra="forbid")

    index: int
    title: str
    content: str


class ChatResponse(BaseModel):
    """Complete non-streaming response returned by POST /api/v1/chat."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    answer: str
    answer_mode: str
    socratic: bool
    decomposed: bool
    subquestions: list[str]
    citations: list[CitationResponse]
    sections: list[AnswerSectionResponse]
    warnings: list[str]
    provider: str
    model: str


def _get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized.",
        )
    return service


def _build_filters(payload: ChatRequest) -> Optional[SearchFilters]:
    filters = SearchFilters(
        document_ids=payload.document_ids,
        file_types=payload.file_types,
        epoch_min=payload.epoch_min,
        epoch_max=payload.epoch_max,
    )
    return None if filters.is_empty() else filters


def _format_sse_frame(event_name: str, data: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat_completion(request: Request, payload: ChatRequest) -> ChatResponse:
    """Execute grounded local retrieval and generate a complete chat answer."""
    service = _get_chat_service(request)
    filters = _build_filters(payload)

    try:
        result = service.chat(
            query=payload.query,
            answer_mode=payload.answer_mode,
            socratic=payload.socratic,
            search_mode=payload.search_mode,
            top_k=payload.top_k,
            filters=filters,
            decompose=payload.decompose,
            use_llm_planner=payload.use_llm_planner,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
    except (LLMUnavailableError, EmbeddingUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMCancelledError as exc:
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST
            if hasattr(status, "HTTP_499_CLIENT_CLOSED_REQUEST")
            else 499,
            detail=str(exc),
        ) from exc
    except LLMGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        query_id=result.query_id,
        query=result.query,
        answer=result.answer,
        answer_mode=result.answer_mode,
        socratic=result.socratic,
        decomposed=result.decomposed,
        subquestions=result.subquestions,
        citations=[CitationResponse(**c.to_dict()) for c in result.citations],
        sections=[AnswerSectionResponse(**sec) for sec in result.sections],
        warnings=result.warnings,
        provider=result.provider,
        model=result.model,
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Stream grounded local chat progress, citations, tokens, sections, and completion via SSE."""
    service = _get_chat_service(request)
    filters = _build_filters(payload)

    event_queue: queue.Queue[object] = queue.Queue()
    cancel_event = threading.Event()

    def _worker() -> None:
        try:
            for event_name, data in service.stream_events(
                query=payload.query,
                answer_mode=payload.answer_mode,
                socratic=payload.socratic,
                search_mode=payload.search_mode,
                top_k=payload.top_k,
                filters=filters,
                decompose=payload.decompose,
                use_llm_planner=payload.use_llm_planner,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                cancel_check=cancel_event.is_set,
            ):
                if cancel_event.is_set():
                    break
                event_queue.put((event_name, data))
        except (LLMUnavailableError, EmbeddingUnavailableError) as exc:
            event_queue.put(("error", {"code": "model_unavailable", "message": str(exc)}))
            event_queue.put(("done", {"status": "error", "error": str(exc)}))
        except Exception as exc:
            event_queue.put(("error", {"code": "internal_error", "message": str(exc)}))
            event_queue.put(("done", {"status": "error", "error": str(exc)}))
        finally:
            event_queue.put(_STREAM_END)

    worker = threading.Thread(target=_worker, name="mindvault-chat-stream", daemon=True)
    worker.start()

    async def _sse_generator() -> AsyncIterator[str]:
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    break

                try:
                    item = event_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue

                if item is _STREAM_END:
                    break

                event_name, data = item  # type: ignore[misc]
                yield _format_sse_frame(str(event_name), data)  # type: ignore[arg-type]
        finally:
            cancel_event.set()
            worker.join(timeout=1.0)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
