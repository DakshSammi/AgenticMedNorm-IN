from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
  document_uid TEXT PRIMARY KEY,
  source_document_id TEXT,
  collection_date TEXT,
  source_json_relpath TEXT,
  source_json_sha256 TEXT,
  status TEXT
);
CREATE TABLE IF NOT EXISTS pages (
  page_uid TEXT PRIMARY KEY,
  document_uid TEXT,
  page_number INTEGER,
  raw_image_relpath TEXT,
  anonymized_image_relpath TEXT,
  lineage_status TEXT,
  status TEXT
);
CREATE TABLE IF NOT EXISTS mentions (
  mention_id TEXT PRIMARY KEY,
  document_uid TEXT,
  page_uid TEXT,
  source_json_path TEXT,
  source_object_index INTEGER,
  status TEXT
);
CREATE TABLE IF NOT EXISTS stage_runs (
  run_id TEXT PRIMARY KEY,
  stage_name TEXT,
  status TEXT,
  started_at TEXT,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  artifact_type TEXT,
  path TEXT,
  sha256 TEXT,
  run_id TEXT,
  status TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  stage_name TEXT,
  stable_input_id TEXT,
  status TEXT,
  priority INTEGER,
  duplicate_group_id TEXT,
  canonical_duplicate_representative TEXT,
  reason TEXT
);
CREATE TABLE IF NOT EXISTS errors (
  error_id TEXT PRIMARY KEY,
  stage_name TEXT,
  stable_input_id TEXT,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT
);
"""


class PipelineStateRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def replace_rows(self, table: str, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
        placeholders = ",".join("?" for _ in columns)
        col_sql = ",".join(columns)
        self.conn.execute(f"DELETE FROM {table}")
        self.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})",
            [[row.get(col) for col in columns] for row in rows],
        )
        self.conn.commit()

    def table_count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])
