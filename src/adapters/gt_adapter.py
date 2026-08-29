import json
from pathlib import Path
from typing import Dict, Any
from src.schemas.raw_extraction import (
    CanonicalRawDoc,
    PatientInformation,
    EncounterInformation,
    RawEntityItem,
    RawMedicationItem,
    RawFollowUp,
    RawLabObservationItem,
    Metadata
)

class GTAdapter:
    @staticmethod
    def from_file(file_path: Path) -> CanonicalRawDoc:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GTAdapter.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CanonicalRawDoc:
        meta_data = data.get("document_metadata", {})
        doc_id = meta_data.get("document_id", "unknown")
        
        entities = data.get("raw_entities", {})
        
        # 1. Patient Information
        p_info = entities.get("patient_information", {})
        patient_info = PatientInformation(
            name=p_info.get("name") or None,
            age=p_info.get("age") or None,
            gender=p_info.get("gender") or None,
            address=p_info.get("address") or None,
            phone=p_info.get("phone") or None,
            patient_identifier=p_info.get("patient_identifier") or None,
            abha_id=p_info.get("abha_id") or None
        )
        
        # 2. Encounter Information
        e_info = entities.get("encounter_information", {})
        encounter_info = EncounterInformation(
            date=e_info.get("date") or None,
            department=e_info.get("department") or None,
            hospital_name=e_info.get("hospital_name") or None,
            doctor_name=e_info.get("doctor_name") or None,
            visit_type=e_info.get("visit_type") or None,
            fees=e_info.get("fees") or None,
            room_or_queue_no=e_info.get("room_or_queue_no") or None
        )
        
        # 3. Simple list string elements -> RawEntityItems
        def parse_entity_list(items) -> list:
            results = []
            if not items:
                return results
            for item in items:
                if isinstance(item, str):
                    results.append(RawEntityItem(
                        raw_text=item,
                        evidence_text=item,
                        page_number=1
                    ))
                elif isinstance(item, dict):
                    results.append(RawEntityItem(
                        raw_text=item.get("raw_text") or item.get("text") or "",
                        evidence_text=item.get("evidence_text") or item.get("raw_text") or "",
                        page_number=item.get("page_number", 1),
                        confidence=item.get("confidence"),
                        section=item.get("section")
                    ))
            return results

        complaints = parse_entity_list(entities.get("complaints_or_diagnosis", []))
        observations = parse_entity_list(entities.get("observations", []))
        procedures = parse_entity_list(entities.get("procedures", []))
        advice = parse_entity_list(entities.get("advice", []))
        allergies = parse_entity_list(entities.get("allergy_mentions", []))
        notes = parse_entity_list(entities.get("other_notes", []))
        
        # 4. Medications
        medications = []
        raw_med_list = entities.get("medications", [])
        for item in raw_med_list:
            if isinstance(item, dict):
                # Build reconstructed raw_line_text if not present
                raw_name = item.get("raw_medication_text") or item.get("raw_name") or ""
                raw_dosage = item.get("raw_dosage_text") or item.get("raw_dosage") or ""
                raw_freq = item.get("raw_frequency_text") or item.get("raw_frequency") or ""
                raw_route = item.get("raw_route_text") or item.get("raw_route") or ""
                raw_duration = item.get("raw_duration_text") or item.get("raw_duration") or ""
                raw_inst = item.get("raw_instruction_text") or item.get("raw_instruction") or ""
                raw_timing = item.get("raw_timing_text") or item.get("raw_timing") or ""
                
                effective_route = raw_route
                if raw_route and raw_route in raw_name:
                    effective_route = ""
                
                parts = [p for p in [raw_name, raw_dosage, effective_route, raw_freq, raw_duration, raw_inst, raw_timing] if p]
                reconstructed_line = " ".join(parts) if parts else ""
                
                medications.append(RawMedicationItem(
                    raw_line_text=item.get("raw_line_text") or reconstructed_line,
                    raw_name=raw_name or None,
                    raw_dosage=raw_dosage or None,
                    raw_route=raw_route or None,
                    raw_frequency=raw_freq or None,
                    raw_duration=raw_duration or None,
                    raw_instruction=raw_inst or None,
                    raw_timing=raw_timing or None,
                    evidence_text=item.get("evidence_text") or item.get("raw_medication_text") or reconstructed_line,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence"),
                    section=item.get("section")
                ))
            elif isinstance(item, str):
                medications.append(RawMedicationItem(
                    raw_line_text=item,
                    raw_name=item,
                    evidence_text=item,
                    page_number=1
                ))
                
        # 5. Follow Up
        follow_up = None
        f_up = entities.get("follow_up")
        if f_up:
            if isinstance(f_up, dict):
                follow_up = RawFollowUp(
                    raw_text=f_up.get("raw_text") or "",
                    date=f_up.get("date") or None,
                    review_after=f_up.get("review_after") or None
                )
            elif isinstance(f_up, str):
                follow_up = RawFollowUp(raw_text=f_up)
                
        # 6. Metadata
        ext_meta = data.get("extraction_metadata", {})
        raw_source_type = meta_data.get("source_type")
        mapped_source_type = None
        if raw_source_type:
            mapped_source_type = raw_source_type.lower().strip()
            mapping = {
                "opd_prescription": "prescription",
                "diagnostic_reports": "diagnostic_report",
            }
            mapped_source_type = mapping.get(mapped_source_type, mapped_source_type)

        metadata = Metadata(
            model_name=meta_data.get("ocr_engine", "manual_annotation"),
            backend_name="manual_annotation",
            schema_version="raw_rx_v2",
            timestamp=meta_data.get("created_at"),
            uncertainty_notes=ext_meta.get("model_notes"),
            document_type=mapped_source_type
        )
        
        # 7. Lab Observations
        lab_observations = []
        raw_lab_list = entities.get("lab_observations", [])
        for item in raw_lab_list:
            if isinstance(item, dict):
                test_name = item.get("test_name") or ""
                result = item.get("result") or ""
                unit = item.get("unit") or ""
                ref_range = item.get("reference_range") or ""
                
                parts = [p for p in [test_name, result, unit, ref_range] if p]
                reconstructed_line = " ".join(parts) if parts else ""
                
                lab_observations.append(RawLabObservationItem(
                    raw_line_text=item.get("raw_line_text") or reconstructed_line,
                    test_name=test_name or None,
                    result=result or None,
                    unit=unit or None,
                    reference_range=ref_range or None,
                    evidence_text=item.get("evidence_text") or item.get("raw_line_text") or reconstructed_line,
                    page_number=item.get("page_number", 1),
                    confidence=item.get("confidence"),
                    section=item.get("section")
                ))
            elif isinstance(item, str):
                lab_observations.append(RawLabObservationItem(
                    raw_line_text=item,
                    test_name=item,
                    evidence_text=item,
                    page_number=1
                ))
                
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
            follow_up=follow_up,
            allergy_mentions=allergies,
            other_notes=notes,
            lab_observations=lab_observations,
            metadata=metadata
        )
