# Frozen Configs

This directory may contain frozen private-run manifests in the working environment. Public release commits should include only configs that do not contain private absolute paths, credentials, patient-level rows, or machine-specific cache locations.

For the public synthetic smoke test, use:

```bash
configs/examples/synthetic_pipeline_config.json
```
