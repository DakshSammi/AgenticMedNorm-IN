import csv
from pathlib import Path
from typing import Dict, Any, List

class HandoffAdapter:
    @staticmethod
    def load_handoff_csv(path: Path) -> List[Dict[str, str]]:
        """Loads a handoff CSV containing OCR outputs."""
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
            
    @staticmethod
    def filter_available(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filters handoff rows to include only those marked successful/available."""
        return [
            row for row in rows 
            if row.get("status") in ("ok", "available")
        ]

    @staticmethod
    def map_to_server1_import(rows: List[Dict[str, str]], manifest: Dict[str, Dict[str, str]], import_root: Path) -> List[Dict[str, Any]]:
        """Maps Server 2 handoff rows to Server 1 import format."""
        mapped = []
        for row in rows:
            doc_id = row["document_id"]
            engine = row.get("engine") or row.get("ocr_engine", "")
            raw_path = row.get("raw_text_path") or row.get("ocr_text_path", "")
            status = row.get("status", "")
            
            raw_abs = import_root / raw_path if raw_path else Path("")
            ok = status in ("ok", "available") and raw_abs.exists()
            doc_info = manifest.get(doc_id, {})
            
            mapped.append({
                "document_id": doc_id,
                "patient_id": doc_info.get("patient_id", doc_id),
                "ocr_engine": engine,
                "ocr_text_path": str(raw_abs) if raw_abs.exists() else "",
                "status": "available" if ok else "missing",
                "runtime": row.get("runtime", ""),
                "env_name": row.get("env_name", ""),
                "markdown_path": str(import_root / row.get("markdown_path", "")) if row.get("markdown_path") else "",
                "layout_json_path": str(import_root / row.get("layout_json_path", "")) if row.get("layout_json_path") else "",
                "pagewise_text_paths": "|".join(str(import_root / p.strip()) for p in row.get("pagewise_text_paths", "").split(";") if p.strip()),
                "notes": row.get("notes", ""),
            })
        return mapped
