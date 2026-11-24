-- MindVault initial relational schema (SQLite)

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    filename TEXT NOT NULL CHECK (length(trim(filename)) > 0),
    file_type TEXT NOT NULL CHECK (length(trim(file_type)) > 0),
    source_path TEXT NOT NULL CHECK (length(trim(source_path)) > 0),
    content_hash TEXT NOT NULL CHECK (length(trim(content_hash)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(metadata_json) >= 2),
    UNIQUE (source_path, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    document_id TEXT NOT NULL
        REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    epoch INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0),
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(metadata_json) >= 2),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_epoch ON chunks (epoch);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks (created_at);

CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    chunk_id TEXT NOT NULL
        REFERENCES chunks (id) ON DELETE CASCADE,
    belief_text TEXT NOT NULL CHECK (length(trim(belief_text)) > 0),
    normalized_text TEXT NOT NULL CHECK (length(trim(normalized_text)) > 0),
    confidence REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    epoch INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'contradicted', 'archived')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_beliefs_chunk_id ON beliefs (chunk_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_normalized_text ON beliefs (normalized_text);
CREATE INDEX IF NOT EXISTS idx_beliefs_epoch ON beliefs (epoch);
CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs (status);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    belief_a_id TEXT NOT NULL
        REFERENCES beliefs (id) ON DELETE CASCADE,
    belief_b_id TEXT NOT NULL
        REFERENCES beliefs (id) ON DELETE CASCADE,
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'dismissed')),
    resolution TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    CHECK (belief_a_id <> belief_b_id),
    UNIQUE (belief_a_id, belief_b_id)
);

CREATE INDEX IF NOT EXISTS idx_contradictions_belief_a_id ON contradictions (belief_a_id);
CREATE INDEX IF NOT EXISTS idx_contradictions_belief_b_id ON contradictions (belief_b_id);
CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions (status);
CREATE INDEX IF NOT EXISTS idx_contradictions_score ON contradictions (score DESC);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    source_id TEXT NOT NULL CHECK (length(trim(source_id)) > 0),
    target_id TEXT NOT NULL CHECK (length(trim(target_id)) > 0),
    edge_type TEXT NOT NULL CHECK (length(trim(edge_type)) > 0),
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0),
    created_at TEXT NOT NULL,
    CHECK (source_id <> target_id),
    UNIQUE (source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source_id ON graph_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target_id ON graph_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_edge_type ON graph_edges (edge_type);

CREATE TABLE IF NOT EXISTS retention_scores (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    chunk_id TEXT NOT NULL UNIQUE
        REFERENCES chunks (id) ON DELETE CASCADE,
    strength REAL NOT NULL DEFAULT 1.0 CHECK (strength >= 0.0),
    retention REAL NOT NULL DEFAULT 1.0
        CHECK (retention >= 0.0 AND retention <= 1.0),
    last_reviewed_at TEXT,
    next_review_at TEXT,
    review_count INTEGER NOT NULL DEFAULT 0 CHECK (review_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_retention_scores_next_review_at ON retention_scores (next_review_at);
CREATE INDEX IF NOT EXISTS idx_retention_scores_retention ON retention_scores (retention);

CREATE TABLE IF NOT EXISTS citations (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    query_id TEXT NOT NULL CHECK (length(trim(query_id)) > 0),
    chunk_id TEXT NOT NULL
        REFERENCES chunks (id) ON DELETE CASCADE,
    citation_type TEXT NOT NULL DEFAULT 'retrieval'
        CHECK (citation_type IN ('retrieval', 'belief', 'brief', 'manual')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_query_id ON citations (query_id);
CREATE INDEX IF NOT EXISTS idx_citations_chunk_id ON citations (chunk_id);
CREATE INDEX IF NOT EXISTS idx_citations_citation_type ON citations (citation_type);

CREATE TABLE IF NOT EXISTS morning_briefs (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    brief_date TEXT NOT NULL UNIQUE CHECK (length(trim(brief_date)) > 0),
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(metadata_json) >= 2),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_morning_briefs_brief_date ON morning_briefs (brief_date);
