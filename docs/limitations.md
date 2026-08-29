# Limitations

The current release should be interpreted as an engineering and automated semantic-audit study, not as a human-adjudicated clinical accuracy study.

Key limitations:

- the corpus comes from a single General Medicine OPD setting
- structured prescription annotations were produced by an automated annotation agent
- independent human semantic adjudication was not completed for this submission
- LLM-as-judge outputs are audit evidence, not ground truth
- the 125-document annotation-model benchmark is a legacy model-selection benchmark with limited per-record/model-alias reconstruction
- benchmark results should not be generalized into a family-wide claim that proprietary models are superior
- India-specific knowledge-base coverage remains incomplete
- RxNorm is U.S.-oriented and may not represent exact Indian product equivalence
- ATC mapping is partial and should be treated as therapeutic-class support only
- the open India candidate layer is useful for recall but is not an authority source
- external site and multi-institution validation remain future work
- `HUMAN_REVIEW` outputs are intentionally unresolved until adjudicated

Future work should add independent expert adjudication, multi-reviewer agreement, calibration of automated semantic auditors against expert references, and an expert-adjudicated public benchmark subset.
