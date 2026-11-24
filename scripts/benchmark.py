"""Local offline benchmark suite for MindVault ingestion, search, decay, and graph ops."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def run_benchmark(*, num_docs: int = 12, queries: int = 10) -> dict[str, object]:
    """Run an offline deterministic benchmark inside an isolated temporary vault."""
    from app.core.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        settings = Settings(
            data_dir=base,
            sqlite_path=base / "bench.db",
            chroma_dir=base / "chroma",
            models_dir=base / "models",
            logs_dir=base / "logs",
            vault_dir=base / "vault",
            scheduler_enabled=False,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            t_ingest_0 = time.perf_counter()
            total_chunks = 0
            for idx in range(num_docs):
                payload = (
                    f"# Domain Topic {idx}\n\n"
                    f"Document {idx} captures local offline engineering notes on SQLite WAL, "
                    f"exponential retention R = exp(-t / S), and hybrid BM25 RRF retrieval "
                    f"across epoch {idx % 4 + 1}.\n\n"
                    f"Secondary paragraph with concept keyword cluster_{idx} and graph edges.\n"
                ).encode()
                resp = client.post(
                    "/api/v1/ingest/file",
                    files={"file": (f"bench_doc_{idx}.md", payload, "text/markdown")},
                    data={"epoch": str((idx % 4) + 1)},
                )
                if resp.status_code != 201:
                    raise RuntimeError(f"Ingestion failed during benchmark: {resp.text}")
                total_chunks += int(resp.json()["chunk_count"])
            ingest_ms = round((time.perf_counter() - t_ingest_0) * 1000.0, 2)

            t_search_0 = time.perf_counter()
            for q_idx in range(queries):
                s_resp = client.post(
                    "/api/v1/search",
                    json={"query": f"SQLite WAL retrieval epoch {q_idx % 4 + 1}", "top_k": 5},
                )
                if s_resp.status_code != 200:
                    raise RuntimeError(f"Search failed during benchmark: {s_resp.text}")
            search_ms = round((time.perf_counter() - t_search_0) * 1000.0, 2)

            t_ret_0 = time.perf_counter()
            ret_resp = client.get("/api/v1/retention", params={"limit": 500})
            if ret_resp.status_code != 200:
                raise RuntimeError(f"Retention check failed: {ret_resp.text}")
            retention_ms = round((time.perf_counter() - t_ret_0) * 1000.0, 2)

            t_dash_0 = time.perf_counter()
            dash_resp = client.get("/api/v1/dashboard/overview")
            if dash_resp.status_code != 200:
                raise RuntimeError(f"Dashboard overview failed: {dash_resp.text}")
            dashboard_ms = round((time.perf_counter() - t_dash_0) * 1000.0, 2)

            t_graph_0 = time.perf_counter()
            graph_resp = client.post("/api/v1/graph/rebuild", json={})
            if graph_resp.status_code != 200:
                raise RuntimeError(f"Graph rebuild failed: {graph_resp.text}")
            graph_ms = round((time.perf_counter() - t_graph_0) * 1000.0, 2)

    return {
        "status": "ok",
        "offline_only": True,
        "documents_ingested": num_docs,
        "chunks_created": total_chunks,
        "queries_executed": queries,
        "timings_ms": {
            "ingestion_total_ms": ingest_ms,
            "ingestion_per_doc_ms": round(ingest_ms / max(1, num_docs), 2),
            "search_total_ms": search_ms,
            "search_avg_ms": round(search_ms / max(1, queries), 2),
            "retention_eval_ms": retention_ms,
            "dashboard_overview_ms": dashboard_ms,
            "graph_rebuild_ms": graph_ms,
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run local MindVault performance benchmarks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a lightweight benchmark configuration suitable for CI checks.",
    )
    args = parser.parse_args(argv)
    docs = 6 if args.quick else 24
    queries = 5 if args.quick else 20
    report = run_benchmark(num_docs=docs, queries=queries)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
