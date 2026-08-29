import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.schemas.raw_extraction import (
    CanonicalRawDoc,
    PatientInformation,
    EncounterInformation,
    RawEntityItem,
    RawMedicationItem,
    Metadata
)

class LegacyPredictionAdapter:
    @staticmethod
    def from_file(file_path: Path) -> CanonicalRawDoc:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = file_path.stem
        return LegacyPredictionAdapter.from_dict(data, doc_id)

    @staticmethod
    def from_dict(data: Dict[str, Any], doc_id: str) -> CanonicalRawDoc:
        # Extract patient_info
        p_info = data.get("patient_info", {})
        
        def safe_str(val) -> Optional[str]:
            if val is None:
                return None
            return str(val).strip() or None
            
        patient_info = PatientInformation(
            name=safe_str(p_info.get("name")),
            age=safe_str(p_info.get("age")),
            gender=safe_str(p_info.get("gender")),
            address=safe_str(p_info.get("address")),
            phone=safe_str(p_info.get("phone")),
            patient_identifier=safe_str(p_info.get("patient_identifier") or p_info.get("weight")),
            abha_id=safe_str(p_info.get("abha_id"))
        )
        
        # Encounter info is empty in legacy predictions, but we initialize it cleanly
        encounter_info = EncounterInformation()
        
        complaints: List[RawEntityItem] = []
        observations: List[RawEntityItem] = []
        medications: List[RawMedicationItem] = []
        procedures: List[RawEntityItem] = []
        advice: List[RawEntityItem] = []
        allergies: List[RawEntityItem] = []
        notes: List[RawEntityItem] = []
        
        # Parse items
        items = data.get("items", [])
        for i, item in enumerate(items):
            field_path = f"items[{i}]"
            med_name = item.get("medicine_name") or item.get("raw_text") or item.get("name") or ""
            category = item.get("category", "medication")
            
            # Map strictly based on explicitly specified category in the prediction JSON, without keyword heuristics
            if category == "procedure":
                procedures.append(RawEntityItem(
                    raw_text=med_name,
                    evidence_text=item.get("evidence_text") or med_name,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence", 1.0),
                    original_category="items",
                    original_field_path=field_path,
                    adapter_transformation_notes=f"Mapped legacy procedure item '{med_name}' based on predicted category."
                ))
            elif category == "advice":
                advice.append(RawEntityItem(
                    raw_text=med_name,
                    evidence_text=item.get("evidence_text") or med_name,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence", 1.0),
                    original_category="items",
                    original_field_path=field_path,
                    adapter_transformation_notes=f"Mapped legacy advice item '{med_name}' based on predicted category."
                ))
            elif category == "observation":
                observations.append(RawEntityItem(
                    raw_text=med_name,
                    evidence_text=item.get("evidence_text") or med_name,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence", 1.0),
                    original_category="items",
                    original_field_path=field_path,
                    adapter_transformation_notes=f"Mapped legacy observation item '{med_name}' based on predicted category."
                ))
            else:
                # Default/medication
                med_name_val = safe_str(item.get("medicine_name") or item.get("name"))
                dosage = safe_str(item.get("dosage"))
                freq = safe_str(item.get("frequency"))
                duration = safe_str(item.get("duration"))
                inst = safe_str(item.get("instructions"))
                raw_t = safe_str(item.get("raw_text"))
                
                raw_line = raw_t or med_name_val
                evidence = raw_t or med_name_val
                
                medications.append(RawMedicationItem(
                    raw_line_text=raw_line,
                    raw_name=med_name_val,
                    raw_dosage=dosage,
                    raw_frequency=freq,
                    raw_duration=duration,
                    raw_instruction=inst,
                    evidence_text=evidence,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence", 1.0),
                    original_category="items",
                    original_field_path=field_path,
                    adapter_transformation_notes=f"Mapped legacy item '{med_name_val}' into CanonicalRawDoc medications structure."
                ))
                
        # Parse clinical notes
        clin_notes = data.get("clinical_notes")
        if clin_notes:
            note_lines = []
            if isinstance(clin_notes, str):
                note_lines = [line.strip() for line in clin_notes.split("\n") if line.strip()]
            elif isinstance(clin_notes, list):
                for item in clin_notes:
                    if isinstance(item, str):
                        note_lines.append(item)
                    elif isinstance(item, dict):
                        note_lines.append(item.get("raw_text") or item.get("text") or str(item))
            
            for idx, line in enumerate(note_lines):
                # Map clinical notes strictly to other_notes, preserving the original predicted category
                notes.append(RawEntityItem(
                    raw_text=line,
                    evidence_text=line,
                    page_number=1,
                    original_category="clinical_notes",
                    original_field_path=f"clinical_notes[{idx}]",
                    adapter_transformation_notes="Mapped legacy clinical note line to other_notes."
                ))
                    
        # Parse metadata
        legacy_meta = data.get("metadata", {})
        metadata = Metadata(
            model_name=legacy_meta.get("model_name") or "Qwen/Qwen2-VL-7B-Instruct",
            model_version="legacy",
            prompt_version="legacy_v1",
            backend_name="legacy",
            processing_time_ms=legacy_meta.get("processing_time_ms", 0.0),
            schema_version="raw_rx_v2",
            timestamp=legacy_meta.get("timestamp"),
            confidence_score=legacy_meta.get("confidence_score")
        )
        
        return CanonicalRawDoc(
            schema_version="raw_rx_v2",
            document_id=doc_id,
            patient_information=patient_info,
            encounter_information=encounter_info,
            complaints_or_diagnosis=complaints,
            observations=observations,
            medications=medications,
            procedures=procedures,
            advice=advice,
            allergy_mentions=allergies,
            other_notes=notes,
            metadata=metadata
        )
