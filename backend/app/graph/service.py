"""Knowledge Graph generation, incremental updates, filtering, and D3 projection."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import Settings
from app.graph.repository import GraphRepository
from app.retrieval.vector_store import ChromaVectorStore

CONCEPT_NODE_TYPES = frozenset({"chunk", "belief"})
CONCEPT_EDGE_TYPES = frozenset(
    {"similar_chunk", "derived_from", "contradicts", "entails", "supersedes"}
)

CITATION_NODE_TYPES = frozenset({"document", "chunk", "query"})
CITATION_EDGE_TYPES = frozenset(
    {"source_of", "adjacent_chunk", "cross_reference", "cites", "co_cited", "similar_chunk"}
)

_MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#?\s]+)(?:[#?][^)]*)?\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]")


def extract_cross_reference_doc_ids(
    text: str,
    *,
    lookup_by_name: dict[str, set[str]],
    self_doc_id: str,
) -> set[str]:
    """Find referenced indexed document IDs from Markdown or wiki links in text."""
    if not text:
        return set()
    candidates: list[str] = []
    for raw_target in _MD_LINK_RE.findall(text):
        candidates.append(Path(raw_target.strip()).name.lower())
    for raw_target in _WIKI_LINK_RE.findall(text):
        candidates.append(Path(raw_target.strip()).name.lower())

    matched_ids: set[str] = set()
    for name in candidates:
        if not name:
            continue
        for key in (name, Path(name).stem.lower()):
            for doc_id in lookup_by_name.get(key, set()):
                if doc_id != self_doc_id:
                    matched_ids.add(doc_id)
    return matched_ids


def _build_filename_lookup(docs: list[dict[str, Any]]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        doc_id = str(doc["id"])
        name = Path(str(doc["filename"])).name.lower()
        stem = Path(name).stem.lower()
        if name:
            lookup[name].add(doc_id)
        if stem:
            lookup[stem].add(doc_id)
    return lookup


class GraphService:
    """Builds and queries D3-compatible Concept and Citation knowledge graphs."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: GraphRepository,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._vector_store = vector_store

    def _compute_similar_chunk_edges(
        self,
        *,
        chunk_ids: list[str],
        threshold: float,
        max_neighbors: int,
        created_at: str,
    ) -> tuple[list[tuple[str, str, str, float, str]], int]:
        if not chunk_ids:
            return [], 0

        vec_map = self._vector_store.get_embeddings(chunk_ids)
        ordered_ids = [cid for cid in chunk_ids if cid in vec_map]
        if not ordered_ids:
            return [], 0

        query_vecs = [vec_map[cid] for cid in ordered_ids]
        neighbor_lists = self._vector_store.query_neighbors(query_vecs, top_k=max_neighbors)

        best_pairs: dict[tuple[str, str], float] = {}
        for idx, src_id in enumerate(ordered_ids):
            neighbors = neighbor_lists[idx] if idx < len(neighbor_lists) else []
            kept = 0
            for tgt_id, sim in neighbors:
                if tgt_id == src_id:
                    continue
                if sim < threshold:
                    continue
                pair = (src_id, tgt_id) if src_id < tgt_id else (tgt_id, src_id)
                prev = best_pairs.get(pair)
                if prev is None or sim > prev:
                    best_pairs[pair] = sim
                kept += 1
                if kept >= max_neighbors:
                    break

        edges = [
            (pair[0], pair[1], "similar_chunk", round(sim, 6), created_at)
            for pair, sim in best_pairs.items()
        ]
        return edges, len(ordered_ids)

    def on_document_indexed(
        self,
        *,
        doc_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Incrementally add citation, cross-reference, and ANN similarity edges for a doc."""
        if not chunks:
            return {"source_of": 0, "adjacent_chunk": 0, "cross_reference": 0, "similar_chunk": 0}

        now_iso = datetime.now(timezone.utc).isoformat()
        edges: list[tuple[str, str, str, float, str]] = []
        ordered = sorted(chunks, key=lambda c: int(c["chunk_index"]))

        for idx, chunk in enumerate(ordered):
            cid = str(chunk["id"])
            edges.append((doc_id, cid, "source_of", 1.0, now_iso))
            if idx + 1 < len(ordered):
                next_cid = str(ordered[idx + 1]["id"])
                edges.append((cid, next_cid, "adjacent_chunk", 0.85, now_iso))

        all_docs = self._repo.list_indexed_documents_brief()
        lookup = _build_filename_lookup(all_docs)
        xref_count = 0
        for chunk in ordered:
            cid = str(chunk["id"])
            targets = extract_cross_reference_doc_ids(
                str(chunk.get("content", "")),
                lookup_by_name=lookup,
                self_doc_id=doc_id,
            )
            for tgt_doc_id in targets:
                edges.append((cid, tgt_doc_id, "cross_reference", 0.90, now_iso))
                xref_count += 1

        # Also check existing chunks for outgoing links targeting the newly added filename.
        new_doc_lookup = _build_filename_lookup([{"id": doc_id, "filename": filename}])
        for existing_chunk in self._repo.list_indexed_chunks_for_rebuild():
            if existing_chunk["document_id"] == doc_id:
                continue
            targets = extract_cross_reference_doc_ids(
                str(existing_chunk.get("content", "")),
                lookup_by_name=new_doc_lookup,
                self_doc_id=str(existing_chunk["document_id"]),
            )
            if doc_id in targets:
                edges.append((str(existing_chunk["id"]), doc_id, "cross_reference", 0.90, now_iso))
                xref_count += 1

        sim_edges, _embedded = self._compute_similar_chunk_edges(
            chunk_ids=[str(c["id"]) for c in ordered],
            threshold=self._settings.graph_similarity_threshold,
            max_neighbors=self._settings.graph_max_neighbors_per_node,
            created_at=now_iso,
        )
        edges.extend(sim_edges)
        self._repo.upsert_edges(edges)

        return {
            "source_of": len(ordered),
            "adjacent_chunk": max(0, len(ordered) - 1),
            "cross_reference": xref_count,
            "similar_chunk": len(sim_edges),
        }

    def remove_entities(self, entity_ids: list[str]) -> int:
        """Remove all graph edges referencing deleted documents, chunks, or beliefs."""
        return self._repo.delete_edges_for_entities(entity_ids)

    def rebuild_graph(
        self,
        *,
        similarity_threshold: Optional[float] = None,
        max_neighbors_per_node: Optional[int] = None,
    ) -> dict[str, Any]:
        """Rebuild Concept and Citation edges cleanly from SQLite and ChromaDB vectors."""
        eff_threshold = (
            self._settings.graph_similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )
        eff_k = (
            self._settings.graph_max_neighbors_per_node
            if max_neighbors_per_node is None
            else int(max_neighbors_per_node)
        )

        self._repo.delete_edges_by_types(
            [
                "similar_chunk",
                "source_of",
                "adjacent_chunk",
                "cross_reference",
                "derived_from",
                "cites",
                "co_cited",
                "entails",
                "contradicts",
                "supersedes",
            ]
        )
        self._repo.prune_dangling_edges()

        now_iso = datetime.now(timezone.utc).isoformat()
        all_docs = self._repo.list_indexed_documents_brief()
        all_chunks = self._repo.list_indexed_chunks_for_rebuild()
        lookup = _build_filename_lookup(all_docs)

        edges: list[tuple[str, str, str, float, str]] = []
        counts: dict[str, int] = defaultdict(int)

        chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in all_chunks:
            chunks_by_doc[str(chunk["document_id"])].append(chunk)

        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_chunks.sort(key=lambda c: int(c["chunk_index"]))
            for idx, chunk in enumerate(doc_chunks):
                cid = str(chunk["id"])
                edges.append((doc_id, cid, "source_of", 1.0, str(chunk["created_at"])))
                counts["source_of"] += 1

                if idx + 1 < len(doc_chunks):
                    next_cid = str(doc_chunks[idx + 1]["id"])
                    edges.append((cid, next_cid, "adjacent_chunk", 0.85, str(chunk["created_at"])))
                    counts["adjacent_chunk"] += 1

                targets = extract_cross_reference_doc_ids(
                    str(chunk["content"]),
                    lookup_by_name=lookup,
                    self_doc_id=doc_id,
                )
                for tgt_doc_id in targets:
                    edges.append((cid, tgt_doc_id, "cross_reference", 0.90, now_iso))
                    counts["cross_reference"] += 1

        rel_data = self._repo.list_beliefs_and_relations_for_rebuild()
        for belief in rel_data["beliefs"]:
            bid = str(belief["id"])
            edges.append(
                (
                    bid,
                    str(belief["chunk_id"]),
                    "derived_from",
                    float(belief["confidence"]),
                    str(belief["created_at"]),
                )
            )
            counts["derived_from"] += 1
            if belief["superseded_by_id"]:
                edges.append(
                    (
                        str(belief["superseded_by_id"]),
                        bid,
                        "supersedes",
                        1.0,
                        str(belief["updated_at"] or belief["created_at"]),
                    )
                )
                counts["supersedes"] += 1

        for ent in rel_data["entailments"]:
            edges.append(
                (
                    str(ent["source_belief_id"]),
                    str(ent["target_belief_id"]),
                    "entails",
                    float(ent["entailment_score"]),
                    str(ent["created_at"]),
                )
            )
            counts["entails"] += 1

        for con in rel_data["contradictions"]:
            edges.append(
                (
                    str(con["belief_a_id"]),
                    str(con["belief_b_id"]),
                    "contradicts",
                    float(con["score"]),
                    str(con["created_at"]),
                )
            )
            counts["contradicts"] += 1

        cites_by_query: dict[str, tuple[list[str], str]] = {}
        for cit in rel_data["citations"]:
            qid = str(cit["query_id"])
            cid = str(cit["chunk_id"])
            ts = str(cit["created_at"])
            if qid not in cites_by_query:
                cites_by_query[qid] = ([], ts)
            if cid not in cites_by_query[qid][0]:
                cites_by_query[qid][0].append(cid)

        co_cited_weights: dict[tuple[str, str], tuple[float, str]] = {}
        for qid, (chunk_ids, ts) in cites_by_query.items():
            q_node = f"query:{qid}"
            for cid in chunk_ids:
                edges.append((q_node, cid, "cites", 1.0, ts))
                counts["cites"] += 1
            capped = chunk_ids[:25]
            for i in range(len(capped)):
                for j in range(i + 1, len(capped)):
                    ci, cj = capped[i], capped[j]
                    a, b = (ci, cj) if ci < cj else (cj, ci)
                    prev_w, _ = co_cited_weights.get((a, b), (0.0, ts))
                    co_cited_weights[(a, b)] = (prev_w + 1.0, ts)

        for (a, b), (w, ts) in co_cited_weights.items():
            edges.append((a, b, "co_cited", w, ts))
            counts["co_cited"] += 1

        chunk_ids_all = [str(c["id"]) for c in all_chunks]
        sim_edges, embedded_count = self._compute_similar_chunk_edges(
            chunk_ids=chunk_ids_all,
            threshold=eff_threshold,
            max_neighbors=eff_k,
            created_at=now_iso,
        )
        edges.extend(sim_edges)
        counts["similar_chunk"] = len(sim_edges)

        self._repo.upsert_edges(edges)

        warnings: list[str] = []
        if chunk_ids_all and embedded_count == 0:
            warnings.append(
                "No vector embeddings found in ChromaDB; skipped similar_chunk edges."
            )

        return {
            "status": "rebuilt",
            "documents_indexed": len(all_docs),
            "chunks_indexed": len(all_chunks),
            "chunks_with_vectors": embedded_count,
            "total_edges": sum(counts.values()),
            "edge_counts": dict(counts),
            "warnings": warnings,
        }

    @staticmethod
    def _filter_nodes(
        nodes: list[dict[str, Any]],
        *,
        epoch_min: Optional[int],
        epoch_max: Optional[int],
        document_id: Optional[str],
        search: Optional[str],
        max_nodes: int,
    ) -> list[dict[str, Any]]:
        needle = search.strip().lower() if search and search.strip() else None
        filtered: list[dict[str, Any]] = []
        for node in nodes:
            epoch = int(node.get("epoch", 0))
            if epoch_min is not None and epoch < epoch_min:
                continue
            if epoch_max is not None and epoch > epoch_max:
                continue
            if document_id is not None:
                doc_ids = node.get("document_ids")
                if isinstance(doc_ids, set):
                    if document_id not in doc_ids:
                        continue
                elif node.get("document_id") != document_id:
                    continue
            if needle is not None and needle not in str(node.get("search_blob", "")):
                continue
            filtered.append(node)
            if len(filtered) >= max_nodes:
                break
        return filtered

    @staticmethod
    def _serialize_d3_graph(
        *,
        graph_type: str,
        raw_nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        degree_map: dict[str, int] = {node["id"]: 0 for node in raw_nodes}
        for edge in edges:
            if edge["source"] in degree_map:
                degree_map[edge["source"]] += 1
            if edge["target"] in degree_map:
                degree_map[edge["target"]] += 1

        d3_nodes: list[dict[str, Any]] = []
        orphan_count = 0
        for node in raw_nodes:
            deg = degree_map.get(node["id"], 0)
            is_orphan = deg == 0
            if is_orphan:
                orphan_count += 1
            d3_nodes.append(
                {
                    "id": node["id"],
                    "node_type": node["node_type"],
                    "label": node["label"],
                    "document_id": node.get("document_id"),
                    "filename": node.get("filename"),
                    "file_type": node.get("file_type"),
                    "epoch": int(node.get("epoch", 0)),
                    "degree": deg,
                    "orphan": is_orphan,
                    "metadata": dict(node.get("metadata", {})),
                }
            )

        payload: dict[str, Any] = {
            "graph_type": graph_type,
            "nodes": d3_nodes,
            "edges": edges,
            "stats": {
                "node_count": len(d3_nodes),
                "edge_count": len(edges),
                "orphan_count": orphan_count,
            },
        }
        if extra:
            payload.update(extra)
        return payload

    def _project_graph(
        self,
        *,
        graph_type: str,
        allowed_node_types: frozenset[str],
        allowed_edge_types: frozenset[str],
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        document_id: Optional[str] = None,
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        search: Optional[str] = None,
        min_similarity: Optional[float] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> dict[str, Any]:
        eff_max_nodes = self._settings.graph_max_nodes if max_nodes is None else int(max_nodes)
        eff_max_edges = self._settings.graph_max_edges if max_edges is None else int(max_edges)

        target_node_types = [node_type] if node_type else sorted(allowed_node_types)
        candidate_nodes: list[dict[str, Any]] = []
        for nt in target_node_types:
            if nt not in allowed_node_types:
                continue
            if nt == "document":
                candidate_nodes.extend(self._repo.fetch_document_nodes())
            elif nt == "chunk":
                candidate_nodes.extend(self._repo.fetch_chunk_nodes())
            elif nt == "belief":
                candidate_nodes.extend(self._repo.fetch_belief_nodes())
            elif nt == "query":
                candidate_nodes.extend(self._repo.fetch_query_nodes())

        selected_nodes = self._filter_nodes(
            candidate_nodes,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            document_id=document_id,
            search=search,
            max_nodes=eff_max_nodes,
        )
        node_ids = {n["id"] for n in selected_nodes}

        target_edge_types = (
            [edge_type]
            if (edge_type and edge_type in allowed_edge_types)
            else ([] if edge_type else sorted(allowed_edge_types))
        )
        raw_edges = self._repo.fetch_edges_between(
            node_ids,
            edge_types=target_edge_types,
            limit=eff_max_edges,
        )
        if min_similarity is not None:
            raw_edges = [
                e
                for e in raw_edges
                if e["edge_type"] != "similar_chunk" or e["weight"] >= float(min_similarity)
            ]

        return self._serialize_d3_graph(
            graph_type=graph_type,
            raw_nodes=selected_nodes,
            edges=raw_edges,
        )

    def get_concept_graph(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        document_id: Optional[str] = None,
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        search: Optional[str] = None,
        min_similarity: Optional[float] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return D3-compatible Concept Graph (chunks, beliefs, similarity & NLI edges)."""
        return self._project_graph(
            graph_type="concepts",
            allowed_node_types=CONCEPT_NODE_TYPES,
            allowed_edge_types=CONCEPT_EDGE_TYPES,
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

    def get_citation_graph(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        document_id: Optional[str] = None,
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        search: Optional[str] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return D3-compatible Citation Graph (documents, chunks, queries, references)."""
        return self._project_graph(
            graph_type="citations",
            allowed_node_types=CITATION_NODE_TYPES,
            allowed_edge_types=CITATION_EDGE_TYPES,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            document_id=document_id,
            node_type=node_type,
            edge_type=edge_type,
            search=search,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def get_node_neighborhood(
        self,
        node_id: str,
        *,
        hops: int = 1,
        edge_type: Optional[str] = None,
        max_nodes: int = 100,
    ) -> Optional[dict[str, Any]]:
        """Explore a node's k-hop neighborhood across Concept and Citation edges."""
        resolved_id = self._repo.resolve_node_id(node_id)
        if resolved_id is None:
            return None

        node_map = self._repo.fetch_all_nodes_map()
        if resolved_id not in node_map:
            return None

        eff_hops = max(1, min(int(hops), 3))
        eff_max_nodes = max(1, min(int(max_nodes), self._settings.graph_max_nodes))
        edge_types = [edge_type] if edge_type else None

        visited: set[str] = {resolved_id}
        frontier: set[str] = {resolved_id}

        for _ in range(eff_hops):
            if not frontier or len(visited) >= eff_max_nodes:
                break
            incident = self._repo.fetch_incident_edges(frontier, edge_types=edge_types)
            next_frontier: set[str] = set()
            for edge in incident:
                for endpoint in (edge["source"], edge["target"]):
                    if endpoint in node_map and endpoint not in visited:
                        if len(visited) >= eff_max_nodes:
                            break
                        visited.add(endpoint)
                        next_frontier.add(endpoint)
            frontier = next_frontier

        selected_nodes = [node_map[nid] for nid in visited if nid in node_map]
        selected_nodes.sort(key=lambda n: (n["id"] != resolved_id, n["node_type"], n["id"]))
        edges = self._repo.fetch_edges_between(
            visited,
            edge_types=edge_types,
            limit=self._settings.graph_max_edges,
        )

        return self._serialize_d3_graph(
            graph_type="ego",
            raw_nodes=selected_nodes,
            edges=edges,
            extra={"center_id": resolved_id, "hops": eff_hops},
        )
