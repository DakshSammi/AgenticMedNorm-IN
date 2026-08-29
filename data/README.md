# Data Layout

The public repository does not include real prescription images, anonymized prescription images, or patient-level ground-truth JSONs.

Expected private layout for authorized local runs:

```text
prescription_pipeline_jbhi_ieee/
  raw/<collection-date>/pNNN.<jpg|jpeg|png>
  anonymized/<collection-date>/pNNN.<jpg|jpeg|png>
  ground_truths_json/ground_truth_<collection-date>/pNNN.json
```

Public examples are synthetic:

```text
data/examples/
  annotations/synth_p1.json
  annotations/synth_p2.json
  annotations/synth_p3.json
```

Synthetic examples are for reproducibility smoke tests only and must not be used as clinical evaluation data.
