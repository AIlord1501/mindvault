-- Review history table for spaced repetition outcomes and stability progression

CREATE TABLE IF NOT EXISTS review_history (
    id TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    chunk_id TEXT NOT NULL
        REFERENCES chunks (id) ON DELETE CASCADE,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('forgot', 'hard', 'good', 'easy')),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4),
    previous_stability REAL NOT NULL CHECK (previous_stability > 0.0),
    new_stability REAL NOT NULL CHECK (new_stability > 0.0),
    retention_before REAL NOT NULL
        CHECK (retention_before >= 0.0 AND retention_before <= 1.0),
    elapsed_days REAL NOT NULL CHECK (elapsed_days >= 0.0),
    next_review_at TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_review_history_chunk_id
    ON review_history (chunk_id);
CREATE INDEX IF NOT EXISTS idx_review_history_reviewed_at
    ON review_history (reviewed_at DESC);
