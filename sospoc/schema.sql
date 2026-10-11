-- PoC A schema — a strict subset of ../../proposal.md §5.
-- Column names and types match the real schema so the daemon can open a PoC
-- database unchanged. Omitted here: file_tags, tags_fts, ident_fts, ocr_fts,
-- file_labels, file_vectors_bin, and the EXIF columns.

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS files (
    file_id      INTEGER PRIMARY KEY,
    file_path    TEXT    UNIQUE NOT NULL,
    dev          INTEGER NOT NULL,
    inode        INTEGER NOT NULL,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    mime_type    TEXT    NOT NULL,
    content_hash BLOB,
    state        TEXT    NOT NULL DEFAULT 'pending',  -- pending|embedded|failed
    last_seen    INTEGER NOT NULL,
    indexed_at   INTEGER
);
CREATE INDEX IF NOT EXISTS files_ino   ON files(dev, inode);
CREATE INDEX IF NOT EXISTS files_hash  ON files(content_hash);
CREATE INDEX IF NOT EXISTS files_state ON files(state);

-- Lexical side. A subset of proposal.md §5: path_tokens and ocr_text only
-- (no VLM captions, no zero-shot tags). Same table and column names so the
-- real daemon can open a PoC database unchanged.
CREATE TABLE IF NOT EXISTS file_tags (
    file_id     INTEGER PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
    ocr_text    TEXT NOT NULL DEFAULT '',
    path_tokens TEXT NOT NULL DEFAULT ''
);

CREATE VIRTUAL TABLE IF NOT EXISTS tags_fts USING fts5(
    ocr_text, path_tokens,
    content='file_tags', content_rowid='file_id',
    tokenize='unicode61 remove_diacritics 2'
);

-- External-content FTS5 tables do not self-populate; these are mandatory.
CREATE TRIGGER IF NOT EXISTS file_tags_ai AFTER INSERT ON file_tags BEGIN
  INSERT INTO tags_fts(rowid, ocr_text, path_tokens)
  VALUES (new.file_id, new.ocr_text, new.path_tokens);
END;
CREATE TRIGGER IF NOT EXISTS file_tags_ad AFTER DELETE ON file_tags BEGIN
  INSERT INTO tags_fts(tags_fts, rowid, ocr_text, path_tokens)
  VALUES ('delete', old.file_id, old.ocr_text, old.path_tokens);
END;
CREATE TRIGGER IF NOT EXISTS file_tags_au AFTER UPDATE ON file_tags BEGIN
  INSERT INTO tags_fts(tags_fts, rowid, ocr_text, path_tokens)
  VALUES ('delete', old.file_id, old.ocr_text, old.path_tokens);
  INSERT INTO tags_fts(rowid, ocr_text, path_tokens)
  VALUES (new.file_id, new.ocr_text, new.path_tokens);
END;

-- @vector-index
-- store.py splits on the line above; everything below needs sqlite-vec.

-- float32 in the PoC, not int8. Quantisation is an optimisation; unquantised
-- recall is the baseline that proposal §12.3's overfetch tuning is judged against.
CREATE VIRTUAL TABLE IF NOT EXISTS file_vectors USING vec0(
    file_id   INTEGER PRIMARY KEY,
    embedding float[512] distance_metric=cosine
);
