# Latest Llama Component Ablation Benchmark

This file indexes the device-specific git-tracked latest multi-component Llama ablation snapshots.

- Git-tracked latest snapshots root: `results/llama_component_ablation_prefill/latest`
- Latest-by-device index: [results/llama_component_ablation_prefill/latest/BENCHMARK.md](results/llama_component_ablation_prefill/latest/BENCHMARK.md)
- Standard prompt lengths: `16/32/64/128/256/512/1024/2048/4096/8192`
- Variants: `baseline/replace_ln/replace_attention/replace_rope/replace_activation`
- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.

## Devices

| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |
| --- | --- | --- | --- | --- | --- | --- |
| A100 40G PCIe | `a100_40g_pcie` | [report](results/llama_component_ablation_prefill/latest/a100_40g_pcie/BENCHMARK.md) | [summary](results/llama_component_ablation_prefill/latest/a100_40g_pcie/summary.csv) | [metadata](results/llama_component_ablation_prefill/latest/a100_40g_pcie/metadata.json) | `results/llama_component_ablation_prefill/20260424T082410Z` | `2026-04-24T08:24:18.432008Z` |
