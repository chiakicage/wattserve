# Llama `replace_ln` Latest Snapshots

This directory stores the git-tracked latest canonical Llama `replace_ln` benchmark snapshot for each device.

- Snapshots root: `results/llama_replace_ln_prefill/latest`
- Standard prompt lengths: `16/32/64/128/256/512/1024/2048/4096/8192`
- Publishing is device-scoped and explicit. Timestamped runs do not overwrite these snapshots unless requested.
- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.

## Devices

| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |
| --- | --- | --- | --- | --- | --- | --- |
| A100 40G PCIe | `a100_40g_pcie` | [report](results/llama_replace_ln_prefill/latest/a100_40g_pcie/BENCHMARK.md) | [summary](results/llama_replace_ln_prefill/latest/a100_40g_pcie/summary.csv) | [metadata](results/llama_replace_ln_prefill/latest/a100_40g_pcie/metadata.json) | `results/llama_replace_ln_prefill/20260424T095738Z` | `2026-04-24T09:57:46.366971Z` |
| A100 40G SXM | `a100_40g_sxm` | [report](results/llama_replace_ln_prefill/latest/a100_40g_sxm/BENCHMARK.md) | [summary](results/llama_replace_ln_prefill/latest/a100_40g_sxm/summary.csv) | [metadata](results/llama_replace_ln_prefill/latest/a100_40g_sxm/metadata.json) | `/home/cage/wattserve/results/llama_replace_ln_prefill/20260414T175515Z` | `2026-04-14T17:55:22.057389Z` |
