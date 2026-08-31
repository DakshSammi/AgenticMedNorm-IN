# AgenticMedNorm-IN: Public Demo Dataset

## Subset Contents

This public release contains **10 de-identified prescriptions** with **57 medication mentions** from the AgenticMedNorm-IN cohort.

| Artifact | Path |
|---|---|
| De-identified images (10) | `images/` |
| Medication annotations (57) | `annotations.json` |
| Normalized outputs (57) | `normalization.json` |
| LLM audit results (57) | `llm_audit.json` |
| Checksums | `checksums.sha256` |

## Schema

Each entry in `annotations.json` contains:

- `mention_id` — public AMNIN identifier (e.g. AMNIN_RX_0001_M001)
- `prescription_id` — public AMNIN prescription ID
- `raw_medication_text` — raw text from handwritten prescription
- `raw_strength` — strength if detected
- `raw_dosage_form` — dosage form if detected
- `raw_route` — route if detected
- `raw_frequency` — frequency if detected

Each entry in `normalization.json` contains:

- `mention_id` — same public AMNIN identifier
- `resolution_level` — local | rxnorm | atc
- `normalized_product` — resolved product name
- `normalized_brand_family` — brand family identity
- `normalized_ingredients` — ingredient list
- `normalized_strength` — normalized strength
- `normalized_dosage_form` — normalized dosage form
- `normalized_fdc_structure` — fixed-dose combination structure if applicable
- `RxNorm` — RxCUI if mapped
- `ATC` — ATC code if mapped
- `verification_decision` — ACCEPT | HUMAN_REVIEW | NIL
- `hard_conflicts` — detected conflicts
- `review_reason` — reason for HUMAN_REVIEW if applicable

## Full Dataset

The full AgenticMedNorm-IN dataset contains **893 prescriptions** with **3027 medication mentions** across **782 medication-bearing prescriptions**. To request access to the full dataset, please contact:

**sanju.tiwari@sharda.ac.in**

## License

Apache License 2.0. See [LICENSE](../LICENSE).
