# Output Statistics

Generated on 2026-06-28 from the artifacts currently in this folder.

| Area | Statistic | Value |
|---|---|---:|
| SFT | Final train loss | 1.8410 |
| SFT | Adapter file size | 73,911,112 bytes |
| Preference data | Train rows | 300 |
| Preference data | Eval rows | 50 |
| DPO | Beta | 0.1 |
| DPO | Learning rate | 5e-7 |
| DPO | Epochs | 1 |
| DPO | Adapter file size | 73,911,112 bytes |
| DPO | Endpoint chosen reward | unavailable |
| DPO | Endpoint rejected reward | unavailable |
| DPO | Verifier reward-gap fallback | 0.001 |
| Eval | Side-by-side prompts | 8 |
| Eval | Helpfulness prompts | 4 |
| Eval | Safety prompts | 4 |
| Manual judge | SFT-only wins | 1 |
| Manual judge | SFT+DPO wins | 0 |
| Manual judge | Ties | 7 |
| Screenshots | PNG files present | 5 |
| Benchmark | IFEval/GSM8K/MMLU/AlpacaEval-lite | NaN / unavailable |
| Verification | `python scripts/verify.py` | passes |

## Files used

- `notebooks/01_sft_mini.ipynb`
- `adapters/sft-mini/adapter_model.safetensors`
- `adapters/dpo/adapter_model.safetensors`
- `adapters/dpo/dpo_metrics.json`
- `data/pref/train.parquet`
- `data/pref/eval.parquet`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `data/eval/benchmark_results.json`
- `submission/screenshots/02-sft-loss.png`
- `submission/screenshots/03-dpo-reward-curves.png`
- `submission/screenshots/04-side-by-side-table.png`
- `submission/screenshots/05-manual-rubric.png`
- `submission/screenshots/07-benchmark-comparison.png`
