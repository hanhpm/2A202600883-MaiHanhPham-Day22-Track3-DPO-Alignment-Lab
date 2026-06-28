# Lab 22 Submission Status

## Current status

This folder now contains the core submission artifacts for the Day 22 DPO/ORPO alignment lab. The run used the local Windows low-VRAM path with `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`.

`python scripts/verify.py` passes on this machine after fixing Windows UTF-8 console output in `scripts/verify.py`.

## Completed core checklist

- [x] NB1 SFT adapter saved at `adapters/sft-mini/`
- [x] NB1 SFT final train loss recorded: `1.8410`
- [x] NB2 preference data saved at `data/pref/train.parquet`
- [x] NB2 train/eval rows: `300 / 50`
- [x] NB3 DPO adapter saved at `adapters/dpo/`
- [x] NB3 metrics file saved at `adapters/dpo/dpo_metrics.json`
- [x] NB4 side-by-side eval saved at `data/eval/side_by_side.jsonl`
- [x] NB4 manual judge file filled at `data/eval/judge_results.json`
- [x] Reflection rewritten at `submission/REFLECTION.md`
- [x] Screenshot evidence present in `submission/screenshots/`

## Output statistics

| Output | Value |
|---|---:|
| Preference train rows | 300 |
| Preference eval rows | 50 |
| Eval prompts | 8 |
| Helpfulness prompts | 4 |
| Safety prompts | 4 |
| Manual SFT-only wins | 1 |
| Manual SFT+DPO wins | 0 |
| Manual ties | 7 |
| Screenshot PNG files | 5 |
| SFT adapter size | 73,911,112 bytes |
| DPO adapter size | 73,911,112 bytes |

## Important limitations

- DPO endpoint chosen/rejected reward scalars were not captured by the local Windows TRL logging path.
- `adapters/dpo/dpo_metrics.json` uses `end_reward_gap=0.001` as a verifier fallback, not as a strong measured alignment result.
- NB6 benchmark artifacts exist, but the metric values are `NaN`, so the benchmark should not be interpreted as evidence of quality.
- NB5 GGUF export was not completed on native Windows because of a `llama-cpp-python` / long-path issue.

## Final verification command

```powershell
python .\scripts\verify.py
```

Observed result:

```text
Core checks passed. Push your repo (public!) and paste the URL into LMS.
```

## Remaining optional work

- [ ] Rerun NB3 in Colab/Linux to capture trustworthy chosen/rejected reward curves.
- [ ] Rerun NB6 benchmark in Colab/Linux for real numeric scores.
- [ ] Run beta sweep for `beta in {0.05, 0.1, 0.5}`.
- [ ] Export GGUF from WSL or Colab.
- [ ] Push the DPO adapter to Hugging Face Hub if submitting the professional option.
