from app.trends.trend_analysis import analyze_patient_trends


def test_trend_analysis_increasing():
    resources = [
        {
            "resourceType": "Observation",
            "id": "obs-1",
            "code": {"text": "Hemoglobin A1c"},
            "valueQuantity": {"value": 6.8, "unit": "%"},
            "effectiveDateTime": "2024-01-01T08:00:00Z",
        },
        {
            "resourceType": "Observation",
            "id": "obs-2",
            "code": {"text": "Hemoglobin A1c"},
            "valueQuantity": {"value": 7.5, "unit": "%"},
            "effectiveDateTime": "2024-03-01T08:00:00Z",
        },
    ]
    trends = analyze_patient_trends(resources)
    assert len(trends) == 1
    assert trends[0].trend_label == "increasing"
    assert trends[0].trend_score > 0


def test_trend_analysis_insufficient_data():
    resources = [
        {
            "resourceType": "Observation",
            "id": "obs-1",
            "code": {"text": "Blood pressure"},
            "valueQuantity": {"value": 120, "unit": "mmHg"},
            "effectiveDateTime": "2024-01-01T08:00:00Z",
        }
    ]
    trends = analyze_patient_trends(resources)
    assert trends[0].trend_label == "insufficient data"
