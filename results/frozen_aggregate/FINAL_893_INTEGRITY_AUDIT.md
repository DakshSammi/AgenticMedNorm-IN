# Final 893 Integrity Audit

- generated_at: 2026-08-29T13:29:34.787982+00:00
- FINAL_893_INPUT_COMPLETE: TRUE
- FINAL_893_PIPELINE_COMPLETE: TRUE
- FINAL_893_READY: TRUE
- paid_api_calls_performed: false
- qwen_judge_performed: false
- final893_summary_sha256: `708ef606f17eddea803501cde77c9e8e1d1edb49077e08440eabc560fe0ce09c`

## Counts
- raw: `893`
- anonymized: `893`
- ground_truth_json: `893`
- manifest_rows: `893`
- layer_a_documents: `893`
- layer_a_pages: `893`
- layer_a_mentions: `3027`
- stage2c_trace_rows: `238970`
- stage2c1_union_rows: `181252`
- ranked_candidate_rows: `60540`
- evidence_rows: `60540`
- verification_rows_v1_1: `3027`
- layer_b_rows_v1_1: `3027`
- pending_prescriptions: `156`
- accepted_brand_family_provenance_rows: `2411`
- stage7_smoke_mentions: `20`

## Pending 156 Classification
- new_p868_p893: `26`
- recovered_previously_missing: `130`

## Layer-B v1.1 Decisions
- ACCEPT: `2803`
- HUMAN_REVIEW: `223`
- NIL: `1`

## Layer-B v1.1 Resolution Levels
- CLINICAL_FORMULATION: `1`
- EXACT_LOCAL_PRODUCT: `104`
- INGREDIENT_ONLY: `393`
- LOCAL_BRAND_FAMILY: `2519`
- NO_SUPPORTED_RESOLUTION: `1`
- TERMINOLOGY_ONLY: `9`

## Blockers
- none

## Final Artifacts
- `derived/final893/audit/accepted_brand_family_provenance.csv` sha256 `c920604e66c99d6e89a31fa2f98c4f9e8521eff0e5e3115728fc166d37862cd2`
- `derived/final893/audit/final893_pending_prescriptions.csv` sha256 `2e9fc30df44423f6c6e25c28e3bc8620db8c39358ac6ba80148c91d9ac3c3539`
- `derived/final893/audit/full_dataset_consistency_audit.json` sha256 `45f64d157442ea1ccc5cfb995bf234bfdc39a6c7c477ba5008e5255ccbbcd60b`
- `derived/final893/evidence/evidence_assessments.csv` sha256 `9f2b87a50b72def13aaf5061b6aca2536eee6d59a2d6ffa2d6672f01fac479bf`
- `derived/final893/evidence/evidence_summary.json` sha256 `e3cb85b302b976d1f26a97b821642ca8912c15c154de3664c8e4bed529a4cf30`
- `derived/final893/layer_a/context_bundles.jsonl` sha256 `705dd9c2492ad2763afaa1b3ec2409dbea616528095c45db50f92cb402a863a9`
- `derived/final893/layer_a/layer_a_documents.csv` sha256 `d5e6233f7060e777892b200d343219971937c322ef96aba6dbcc1a8bf1a5803a`
- `derived/final893/layer_a/layer_a_medication_mentions.csv` sha256 `17a97fb4a3e9d1ae64980a83d4544e0eddf2279b484f2dc30af518bab8dfd696`
- `derived/final893/layer_a/layer_a_pages.csv` sha256 `aea75158aea118c99dbbeaee33d93534e8df3816d00c080c8896c66a148f9887`
- `derived/final893/layer_b/layer_b_v1.csv` sha256 `3a15695da5bc9303f2d6d21e2fb68c0fc70398e06280356c56d70dc5ae27f5e9`
- `derived/final893/layer_b/layer_b_v1_1.csv` sha256 `395bf2d3c6773b241c987fdee1eeea1fad2527dfc459c0765b94cafb8296cb4a`
- `derived/final893/manifests/master_document_manifest.csv` sha256 `897ed93a2ef127250f5b0e64617e5e4a7a3ab7367e6627a395f9d51e13065e89`
- `derived/final893/pipeline/stage7_smoke_results.csv` sha256 `c8d721b01af8df719075301a22100af9efaa192a47c4429804109651cc1987a2`
- `derived/final893/pipeline/stage7_smoke_summary.json` sha256 `2ae57c5b4c096be3777043f3134e4ce2d5c3b3369a986ae109c907a51bf2edfb`
- `derived/final893/ranking/ranked_candidates.csv` sha256 `26527cb900038360bb6531b61e237ff40826b29051bdaeb5351027c0a9bc7716`
- `derived/final893/ranking/ranking_summary.json` sha256 `f02ae1c6c18c3844f8d4cc8b043d3a068a0c1bb31e8c61b342f7f9a6870a97c7`
- `derived/final893/reports/CANDIDATE_IDENTITY_NAMESPACE_AUDIT.md` sha256 `7df1a8994f3bbf78210d6debd900caae19e78ba25b7964027a47a8202f79b9fc`
- `derived/final893/reports/FULL_DATASET_CONSISTENCY_AUDIT.md` sha256 `960aa73690e144e3e2acec7b906cb378987e649c4d9241d61b5fdb6222c3eb70`
- `derived/final893/reports/STAGE1_CANONICAL_DATA_LAYER_REPORT.md` sha256 `c78256469ba6d36f3cfa5e280e0c5f3c3f73e74afd3bed38fc7fbd25d996095c`
- `derived/final893/reports/STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md` sha256 `05779dd008b2116d2ab6de0293594515b0cef3983b7a7ebc7c2d4b2f15ae5195`
- `derived/final893/reports/STAGE4_CANDIDATE_RANKING_REPORT.md` sha256 `edc635514a1518daadae14b092d619149bb64d594d332aee5ed7409b0b0df258`
- `derived/final893/reports/STAGE5_EVIDENCE_ASSESSMENT_REPORT.md` sha256 `014b3f13ca60c2d26953bbe36a28facc69a14687dbea17e4a75a21fad83ec183`
- `derived/final893/reports/STAGE6_1_RESOLUTION_AUDIT_REPORT.md` sha256 `ea5cafafda35fb3ea3cde9823525a82d0b1188d9a48ec91ec3f88e0bbccc64cd`
- `derived/final893/reports/STAGE6_VERIFICATION_REPORT.md` sha256 `5cac9614f405b627a63495f3abef467bf46cd65d3f11170d13ee830f1c1f1364`
- `derived/final893/reports/STAGE7_E2E_SMOKE_TEST_REPORT.md` sha256 `d84bf159574106958812a825041c4ee97be1c45d6d0c9ddc7bb4aba141609141`
- `derived/final893/retrieval/stage2c1_branch_traces.csv` sha256 `b1e672f2502b16c5695e4f1f51f194122881962dd88768fb8bdca150a00c9aa8`
- `derived/final893/retrieval/stage2c1_candidate_union.csv` sha256 `4cde5423300a2e226ce3877ce08f8689d428a5e3013d554d7ea0f591f01a1466`
- `derived/final893/retrieval/stage2c1_summary.json` sha256 `a553bc49783dbcafcfc0e1bacfa128fbaf3ccf8870ee013e90fedeb6f60251a8`
- `derived/final893/retrieval/stage2c_branch_traces.csv` sha256 `94db5d4274817280f1f48578eef1bf32012cdf3c7e2b8582a39c50f6f69213aa`
- `derived/final893/retrieval/stage2c_summary.json` sha256 `31a398963dc6eafb42dc6d4f6497669fcce69f2c6edf07eb617b8b363a4fd74a`
- `derived/final893/verification/verification_results.csv` sha256 `892666efd87bfdc26a02e9a5163e5a956cd36309094a2b312a7e07b2cdb2939a`
- `derived/final893/verification/verification_results_v1_1.csv` sha256 `dbd8117967ee9ef8a3623fb625a2cb631fd27936f35b69f2f963a29695a76df4`
- `derived/final893/verification/verification_summary.json` sha256 `8e28a9d71f46ea954836160a4e7f50f007825b8325847aac1a08c23dd614639f`
- `derived/final893/verification/verification_summary_v1_1.json` sha256 `f008b4a22a13e2cbe32c6495fb6a4a284ffa4239a188f8808936a3e002a56afa`
