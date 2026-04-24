# Llama Component Ablation Latest Snapshots

This directory stores the git-tracked latest multi-component Llama ablation snapshot for each device.

- Snapshots root: `results/llama_component_ablation_prefill/latest`
- Standard prompt lengths: `16/32/64/128/256/512/1024/2048/4096/8192`
- Variants: `baseline/replace_ln/replace_attention/replace_rope/replace_activation`
- Publishing is device-scoped and explicit. Timestamped runs do not overwrite these snapshots unless requested.
- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.

## Devices

| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |
| --- | --- | --- | --- | --- | --- | --- |
| A100 40G PCIe | `a100_40g_pcie` | [report](results/llama_component_ablation_prefill/latest/a100_40g_pcie/BENCHMARK.md) | [summary](results/llama_component_ablation_prefill/latest/a100_40g_pcie/summary.csv) | [metadata](results/llama_component_ablation_prefill/latest/a100_40g_pcie/metadata.json) | `results/llama_component_ablation_prefill/20260424T082410Z` | `2026-04-24T08:24:18.432008Z` |
