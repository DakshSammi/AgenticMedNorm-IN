#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATHS = [
    "README.md",
    ".gitignore",
    ".env.example",
    "CITATION.cff",
    "LICENSE_REQUIRED",
    "requirements.txt",
    "src",
    "scripts/run_pipeline.py",
    "scripts/release_scan.py",
    "scripts/reproduce_paper_artifacts.py",
    "release_manifest.json",
    "configs/examples",
    "configs/evaluation/llm_judge_output_schema.json",
    "configs/evidence",
    "configs/ranking",
    "configs/verification",
    "configs/frozen/README.md",
    "configs/frozen/evaluation_final_893_manifest.json",
    "configs/frozen/open_judge_qwen_v2_final_762_protocol.json",
    "configs/frozen/open_judge_gptoss_v1_calibration51_protocol.json",
    "docs/architecture.md",
    "docs/dataset.md",
    "docs/knowledge_resources.md",
    "docs/annotation_benchmark.md",
    "docs/evaluation.md",
    "docs/expert_validation.md",
    "docs/semantic_audit.md",
    "docs/reproducibility.md",
    "docs/data_availability.md",
    "docs/limitations.md",
    "docs/PUBLIC_RELEASE_AUDIT.md",
    "docs/RELEASE_AUDIT.md",
    "docs/FINAL_RELEASE_CHECKLIST.md",
    "data/README.md",
    "data/examples",
    "data/examples_synthetic",
    "data/schemas",
    "paper_artifacts/accounting/PAPER_CLAIM_REGISTRY_FINAL.csv",
    "paper_artifacts/accounting/PAPER_NUMBER_DICTIONARY_FINAL.json",
    "paper_artifacts/benchmarking/annotation_model_leaderboard_reproduced.csv",
    "paper_artifacts/final_metrics/final_metrics.json",
    "paper_artifacts/tables/TABLE_MANIFEST_FINAL.csv",
    "paper_artifacts/tables/table_annotation_model_benchmark_full.csv",
    "paper_artifacts/tables/table_annotation_model_benchmark_full.tex",
    "paper_artifacts/tables/table_annotation_model_benchmark_main.csv",
    "paper_artifacts/tables/table_annotation_model_benchmark_main.tex",
    "paper_artifacts/tables/table_final_corpus_accounting.csv",
    "paper_artifacts/tables/table_final_corpus_accounting.tex",
    "paper_artifacts/tables/table_knowledge_resources.csv",
    "paper_artifacts/tables/table_knowledge_resources.tex",
    "paper_artifacts/tables/table_pipeline_disposition.csv",
    "paper_artifacts/tables/table_pipeline_disposition.tex",
    "paper_artifacts/tables/table_qwen_by_fdc.csv",
    "paper_artifacts/tables/table_qwen_by_resolution.csv",
    "paper_artifacts/tables/table_qwen_by_semantic_id.csv",
    "paper_artifacts/tables/table_qwen_by_verification.csv",
    "paper_artifacts/tables/table_qwen_semantic_audit.csv",
    "paper_artifacts/tables/table_qwen_semantic_audit.tex",
    "paper_artifacts/tables/table_resolution_levels.csv",
    "paper_artifacts/tables/table_resolution_levels.tex",
    "paper_artifacts/tables/table_semantic_auditor_concordance.csv",
    "paper_artifacts/tables/table_semantic_auditor_concordance.tex",
    "paper_artifacts/tables/table_semantic_identifier_coverage.csv",
    "paper_artifacts/tables/table_semantic_identifier_coverage.tex",
    "paper_artifacts/figures/FIGURE_MANIFEST_FINAL.csv",
    "paper_artifacts/figures/FIGURE_RECOMMENDATIONS_FOR_8_PAGE_JBHI.md",
    "paper_artifacts/figures/fig01_six_agent_architecture",
    "paper_artifacts/figures/fig02_study_cohort_flow",
    "paper_artifacts/figures/fig03_annotation_model_benchmark",
    "paper_artifacts/figures/fig04_verification_distribution",
    "paper_artifacts/figures/fig05_resolution_level_distribution",
    "paper_artifacts/figures/fig06_semantic_identifier_coverage",
    "paper_artifacts/figures/fig07_qwen_semantic_audit",
    "paper_artifacts/figures/fig08_qwen_by_resolution",
    "paper_artifacts/figures/fig09_qwen_by_verification",
    "paper_artifacts/figures/fig10_intermodel_concordance",
    "paper_artifacts/figures/fig11_knowledge_resource_coverage",
    "paper_artifacts/figures/fig12_mentions_per_prescription",
    "paper_artifacts/figures/fig13_surface_frequency_long_tail",
    "paper_artifacts/figures/fig14_formulation_characteristics",
    "paper_artifacts/figures/fig15_normalization_case_study",
    "paper_artifacts/figures/fig16_provenance_completeness",
    "paper_artifacts/PAPER_WRITER_HANDOFF_FINAL.md",
    "paper_artifacts/PAPER_WRITER_HANDOFF_FINAL.json",
    "paper_artifacts/CURRENT_MANUSCRIPT_CHANGE_LIST.md",
    "paper_artifacts/DATA_AVAILABILITY_WORDING.md",
    "paper_artifacts/LIMITATIONS_WORDING.md",
    "paper_artifacts/FUTURE_WORK_WORDING.md",
    "paper_artifacts/paper_writer_bundle",
    "paper_artifacts/paper_writer_bundle_manifest.json",
    "paper_artifacts/reproduction_summary.json",
    "paper_artifacts/README.md",
    "knowledge/reports/PAPER_SOURCE_IMPLEMENTATION_MATRIX.csv",
    "results/README.md",
    "results/tables",
    "results/figures",
    "results/frozen_aggregate",
    "tests",
    "supplementary",
]
SECRET_PATTERNS = {
    "api_key": re.compile(r"(?i)(OPENAI_API_KEY|ANNOTATION_API_KEY|API_KEY)\s*[:=]\s*['\"]?(sk-|AIza|ghp_|glpat-|xox[baprs]-)[^'\"\n#]+"),
    "bearer": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    "openai_secret_key": re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}"),
    "password": re.compile(r"(?i)\bpassword\s*[:=]\s*[^#\n]+"),
    "token": re.compile(r"(?i)\btoken\s*[:=]\s*['\"]?(ghp_|glpat-|xox[baprs]-)[^'\"\n#]+"),
    "private_key": re.compile(r"BEGIN (RSA|OPENSSH|EC|DSA)? ?PRIVATE KEY"),
}
PRIVATE_PATTERNS = {
    "server_absolute_path": re.compile(r"/mnt/mnfas[_A-Za-z0-9/.-]*"),
    "private_user_path": re.compile(r"/Data/user-data/[^\\s'\"`]+"),
}
PATH_PATTERNS = {
    "hardcoded_mnt_path": re.compile(r"/mnt/mnfas[_A-Za-z0-9/.-]*"),
    "private_ip": re.compile(r"\b10\.(10|11)\.\d{1,3}\.\d{1,3}\b"),
}
BINARY_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf", ".parquet", ".faiss", ".index", ".sqlite", ".gz", ".zip", ".pyc"}


def public_files() -> list[Path]:
    files: list[Path] = []
    for item in PUBLIC_PATHS:
        path = ROOT / item
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(
        {
            p
            for p in files
            if "__pycache__" not in p.parts
            and ".pytest_cache" not in p.parts
            and not p.name.endswith(".pyc")
        }
    )


def scan(patterns: dict[str, re.Pattern[str]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in public_files():
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if path.name == "release_scan.py" and ("re.compile" in line or "PRIVATE_PATTERNS" in line or "PATH_PATTERNS" in line):
                continue
            for category, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "file": path.relative_to(ROOT).as_posix(),
                            "line": str(line_no),
                            "category": category,
                        }
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["secrets", "private-data", "paths", "all"], default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = []
    if args.check in {"secrets", "all"}:
        checks.append(("secrets", scan(SECRET_PATTERNS)))
    if args.check in {"private-data", "all"}:
        checks.append(("private-data", scan(PRIVATE_PATTERNS)))
    if args.check in {"paths", "all"}:
        checks.append(("paths", scan(PATH_PATTERNS)))

    payload = {name: findings for name, findings in checks}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, findings in checks:
            print(f"{name}: {'PASS' if not findings else 'FAIL'}")
            for item in findings:
                print(f"{item['file']}:{item['line']} {item['category']}")
    return 0 if all(not findings for _, findings in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
