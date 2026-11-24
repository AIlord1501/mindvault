"""REST API endpoints for D3-compatible Concept and Citation knowledge graphs."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.graph.service import GraphService

router = APIRouter(tags=["graph"])


class D3NodeResponse(BaseModel):
    """D3-compatible node payload (orphan nodes always retained)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node_type: str
    label: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    epoch: int
    degree: int
    orphan: bool
    metadata: dict[str, Any]


class D3EdgeResponse(BaseModel):
    """D3-compatible edge payload keyed by node ID strings."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    edge_type: str
    weight: float
    created_at: str


class GraphStatsResponse(BaseModel):
    """Summary counters for a projected graph."""

    model_config = ConfigDict(extra="forbid")

    node_count: int
    edge_count: int
    orphan_count: int


class GraphResponse(BaseModel):
    """Full D3-compatible Concept or Citation graph response."""

    model_config = ConfigDict(extra="forbid")

    graph_type: str
    nodes: list[D3NodeResponse]
    edges: list[D3EdgeResponse]
    stats: GraphStatsResponse


class NodeSubgraphResponse(GraphResponse):
    """Ego-network exploration response centered on a node ID."""

    center_id: str
    hops: int


class RebuildGraphRequest(BaseModel):
    """Optional overrides when rebuilding knowledge graph edges."""

    model_config = ConfigDict(extra="forbid")

    similarity_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    max_neighbors_per_node: Optional[int] = Field(default=None, ge=1, le=50)


class RebuildGraphResponse(BaseModel):
    """Structured summary returned after rebuilding graph edges."""

    model_config = ConfigDict(extra="forbid")

    status: str
    documents_indexed: int
    chunks_indexed: int
    chunks_with_vectors: int
    total_edges: int
    edge_counts: dict[str, int]
    warnings: list[str]


def _get_graph_service(request: Request) -> GraphService:
    service = getattr(request.app.state, "graph_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service is not initialized.",
        )
    return service


def _validate_epochs(epoch_min: Optional[int], epoch_max: Optional[int]) -> None:
    if epoch_min is not None and epoch_max is not None and epoch_min > epoch_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="epoch_min must be less than or equal to epoch_max.",
        )


@router.get("/graph/concepts", response_model=GraphResponse)
def get_concept_graph(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    document_id: Optional[str] = Query(default=None, min_length=1),
    node_type: Optional[str] = Query(default=None, pattern="^(chunk|belief)$"),
    edge_type: Optional[str] = Query(
        default=None,
        pattern="^(similar_chunk|derived_from|contradicts|entails|supersedes)$",
    ),
    search: Optional[str] = Query(default=None, min_length=1, max_length=200),
    min_similarity: Optional[float] = Query(default=None, gt=0.0, le=1.0),
    max_nodes: Optional[int] = Query(default=None, ge=1, le=5000),
    max_edges: Optional[int] = Query(default=None, ge=1, le=20000),
) -> GraphResponse:
    """Return filtered D3-compatible Concept Graph including orphan nodes."""
    _validate_epochs(epoch_min, epoch_max)
    service = _get_graph_service(request)
    payload = service.get_concept_graph(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        document_id=document_id,
        node_type=node_type,
        edge_type=edge_type,
        search=search,
        min_similarity=min_similarity,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    return GraphResponse(**payload)


@router.get("/graph/citations", response_model=GraphResponse)
def get_citation_graph(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    document_id: Optional[str] = Query(default=None, min_length=1),
    node_type: Optional[str] = Query(default=None, pattern="^(document|chunk|query)$"),
    edge_type: Optional[str] = Query(
        default=None,
        pattern="^(source_of|adjacent_chunk|cross_reference|cites|co_cited|similar_chunk)$",
    ),
    search: Optional[str] = Query(default=None, min_length=1, max_length=200),
    max_nodes: Optional[int] = Query(default=None, ge=1, le=5000),
    max_edges: Optional[int] = Query(default=None, ge=1, le=20000),
) -> GraphResponse:
    """Return filtered D3-compatible Citation Graph including orphan nodes."""
    _validate_epochs(epoch_min, epoch_max)
    service = _get_graph_service(request)
    payload = service.get_citation_graph(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        document_id=document_id,
        node_type=node_type,
        edge_type=edge_type,
        search=search,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    return GraphResponse(**payload)


@router.get("/graph/node/{node_id}", response_model=NodeSubgraphResponse)
def get_graph_node(
    request: Request,
    node_id: str,
    hops: int = Query(default=1, ge=1, le=3),
    edge_type: Optional[str] = Query(default=None, min_length=1, max_length=64),
    max_nodes: int = Query(default=100, ge=1, le=1000),
) -> NodeSubgraphResponse:
    """Explore a node's k-hop neighborhood in D3 format."""
    service = _get_graph_service(request)
    payload = service.get_node_neighborhood(
        node_id,
        hops=hops,
        edge_type=edge_type,
        max_nodes=max_nodes,
    )
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph node '{node_id}' not found.",
        )
    return NodeSubgraphResponse(**payload)


@router.post("/graph/rebuild", response_model=RebuildGraphResponse)
def rebuild_graph(
    request: Request,
    payload: Optional[RebuildGraphRequest] = Body(default=None),
) -> RebuildGraphResponse:
    """Rebuild Concept and Citation graph edges from SQLite and ChromaDB vectors."""
    service = _get_graph_service(request)
    opts = payload or RebuildGraphRequest()
    summary = service.rebuild_graph(
        similarity_threshold=opts.similarity_threshold,
        max_neighbors_per_node=opts.max_neighbors_per_node,
    )
    return RebuildGraphResponse(**summary)
