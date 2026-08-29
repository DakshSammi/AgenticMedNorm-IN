from pathlib import Path

from rebuild.run_stage1_layer_a import ROOT, load_json, resolve_project_path


def test_stage1_resolves_historical_relative_annotation_path():
    relpath = "prescription_pipeline_jbhi_ieee/ground_truths_json/ground_truth_30-06-2026/p1.json"
    resolved = resolve_project_path(relpath)
    assert resolved == ROOT / relpath
    assert load_json(resolved)


def test_stage1_resolves_recovered_relative_annotation_path():
    relpath = "prescription_pipeline_jbhi_ieee/ground_truths_json/ground_truth_11-08-2026/p108.json"
    resolved = resolve_project_path(relpath)
    assert resolved == ROOT / relpath
    assert load_json(resolved)


def test_stage1_resolves_new_final893_relative_annotation_path():
    relpath = "prescription_pipeline_jbhi_ieee/ground_truths_json/ground_truth_27-08-2026/p868.json"
    resolved = resolve_project_path(relpath)
    assert resolved == ROOT / relpath
    assert load_json(resolved)


def test_stage1_preserves_absolute_annotation_path():
    absolute = ROOT / "prescription_pipeline_jbhi_ieee/ground_truths_json/ground_truth_27-08-2026/p893.json"
    resolved = resolve_project_path(absolute)
    assert resolved == absolute
    assert load_json(resolved)


def test_stage1_resolver_does_not_duplicate_project_root():
    absolute = Path(ROOT / "prescription_pipeline_jbhi_ieee/ground_truths_json/ground_truth_30-06-2026/p1.json")
    assert str(resolve_project_path(absolute)).count(str(ROOT)) == 1
