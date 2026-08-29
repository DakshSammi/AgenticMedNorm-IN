from app.ingestion.reader import ingest_input


def test_ingest_plain_text():
    resources, entities, provenance, warnings = ingest_input("plain_text", "Patient has hypertension.", file_name="note.txt")
    assert len(resources) == 1
    assert resources[0]["resourceType"] == "DocumentReference"
    assert entities[0]["type"] == "text_document"
    assert provenance[0]["resource_type"] == "DocumentReference"
    assert not warnings


def test_ingest_json_bundle():
    import json

    payload = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "patient-123"}}
        ]
    }
    resources, entities, provenance, warnings = ingest_input("fhir_json", content=json.dumps(payload), file_name="bundle.json")
    assert len(resources) == 1
    assert resources[0]["resourceType"] == "Patient"
    assert provenance[0]["source"] == "bundle.json"
