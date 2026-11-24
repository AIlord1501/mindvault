-- Belief traceability, embeddings, and NLI relationship scores

ALTER TABLE beliefs
    ADD COLUMN superseded_by_id TEXT REFERENCES beliefs (id) ON DELETE SET NULL;
ALTER TABLE beliefs
    ADD COLUMN context_note TEXT;
ALTER TABLE beliefs
    ADD COLUMN embedding_json TEXT;
ALTER TABLE beliefs
    ADD COLUMN updated_at TEXT;

CREATE INDEX IF NOT EXISTS idx_beliefs_superseded_by_id ON beliefs (superseded_by_id);

ALTER TABLE contradictions
    ADD COLUMN entailment_score REAL NOT NULL DEFAULT 0.0
        CHECK (entailment_score >= 0.0 AND entailment_score <= 1.0);
ALTER TABLE contradictions
    ADD COLUMN neutral_score REAL NOT NULL DEFAULT 0.0
        CHECK (neutral_score >= 0.0 AND neutral_score <= 1.0);
ALTER TABLE contradictions
    ADD COLUMN similarity_score REAL NOT NULL DEFAULT 0.0
        CHECK (similarity_score >= 0.0 AND similarity_score <= 1.0);
ALTER TABLE contradictions
    ADD COLUMN resolution_note TEXT;

CREATE TABLE IF NOT EXISTS belief_relations (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    source_belief_id TEXT NOT NULL
        REFERENCES beliefs (id) ON DELETE CASCADE,
    target_belief_id TEXT NOT NULL
        REFERENCES beliefs (id) ON DELETE CASCADE,
    label TEXT NOT NULL
        CHECK (label IN ('contradiction', 'entailment', 'neutral')),
    contradiction_score REAL NOT NULL
        CHECK (contradiction_score >= 0.0 AND contradiction_score <= 1.0),
    entailment_score REAL NOT NULL
        CHECK (entailment_score >= 0.0 AND entailment_score <= 1.0),
    neutral_score REAL NOT NULL
        CHECK (neutral_score >= 0.0 AND neutral_score <= 1.0),
    similarity_score REAL NOT NULL DEFAULT 0.0
        CHECK (similarity_score >= 0.0 AND similarity_score <= 1.0),
    created_at TEXT NOT NULL,
    CHECK (source_belief_id <> target_belief_id),
    UNIQUE (source_belief_id, target_belief_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_relations_source ON belief_relations (source_belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_relations_target ON belief_relations (target_belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_relations_label ON belief_relations (label);
