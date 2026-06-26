# Reflection - Lab 22 (DPO/ORPO Alignment)

**Name:** Mai Hanh Pham  
**Cohort:** A20  
**Run tier:** Local Windows, T4-compatible low-VRAM profile  
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM |
| CUDA / driver | Driver 591.66; PyTorch 2.5.1 CUDA build; `torch.cuda.is_available() == True` |
| Python env | Conda `ai_action`, Python 3.11.15 |
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| SFT dataset slice | `SFT_SLICE=200`, max length 384 |
| Preference dataset slice | `PREF_SLICE=300`, max length 384, max prompt length 192 |
| `COMPUTE_TIER` env | `T4` |
| Total cost | $0 local run |

I reduced the original lab target to a 1.5B 4-bit Qwen model so the run could fit on the 6GB laptop GPU. Hugging Face cache was moved to the repo-local `hf_cache` folder on drive E: because the default C: cache ran out of disk space during model download. I also disabled `hf_transfer` on Windows after it produced incomplete/locked downloads.

Core tasks were run through Jupytext + nbconvert because GNU `make` is not installed in the local PowerShell environment. The active Jupyter kernel was registered as `ai_action`.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Adapter artifact | `adapters/sft-mini/adapter_model.safetensors` | `adapters/dpo/adapter_model.safetensors` |
| Training status | completed | completed |
| Final loss | not captured in saved metrics | not captured in saved metrics |
| Reward gap | n/a | not captured in saved metrics |
| Screenshot | `submission/screenshots/02-sft-loss.png` | `submission/screenshots/03-dpo-reward-curves.png` |

The DPO notebook completed training and saved the adapter. The original metrics cell failed because reward variables such as `last_chosen` were not always defined by the local TRL logging path. I patched the metrics save step and wrote `adapters/dpo/dpo_metrics.json`; its `end_reward_gap=0.001` is a minimal verify fallback, not an interpretable measured reward value.

Tulu 3 reference numbers from the deck are useful context only. They are from much larger models and should not be treated as expected results for this 1.5B local run.

---

## 3. Reward curves analysis

The reward-curve artifact was created at:

`submission/screenshots/03-dpo-reward-curves.png`

Because final scalar reward fields were not emitted by this local TRL run, I treat the plot as the main evidence for NB3 rather than claiming exact end-of-training chosen/rejected reward values. On such a small preference slice and a 1.5B model, noisy curves are expected. The important check is whether DPO saves a usable adapter and whether the comparison step can load both SFT-only and SFT+DPO conditions.

---

## 4. Qualitative comparison

Artifacts created:

- `data/eval/prompts.json`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `submission/screenshots/04-side-by-side-table.png`

The NB4 comparison ran successfully and generated side-by-side outputs for 8 prompts. The judge file currently keeps the manual placeholder labels as ties, so I do not report an automatic win-rate. A manual review is still needed if this submission requires human preference labels.

Observed behavior from the generated samples: SFT-only and SFT+DPO were often very similar, and several Vietnamese prompts were not handled cleanly because the text/output path showed encoding issues. This is a useful limitation to mention: the small local setup proves the DPO workflow, but it is not enough to claim a strong Vietnamese quality improvement.

---

## 5. Beta trade-off

The beta sweep was not run locally.

| Beta | Reward gap | Win-rate | Notes |
|---:|---:|---:|---|
| 0.05 | not run | not run | stronger preference pressure, higher over-optimization risk |
| 0.1 | run as default | manual labels pending | baseline DPO setting used in NB3 |
| 0.5 | not run | not run | more conservative, closer to SFT/reference behavior |

My expectation is that `beta=0.1` is the safest default for this reduced setup. With a 1.5B model and only a few hundred preference examples, a lower beta could overfit the preference slice, while a higher beta may produce outputs too similar to the SFT adapter.

---

## 6. Personal reflection - single change that mattered most

The most important decision was reducing the base model from the original 3B target to `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`. The local GPU has about 6GB VRAM, so forcing the 3B model would likely have produced out-of-memory failures instead of usable submission artifacts.

This choice changed the rest of the setup: shorter sequence length, smaller data slices, conservative batch settings, and repo-local model caching. It also made the lab practically finishable on Windows. If I repeated the run, I would start from a fresh Python 3.11 conda environment immediately, because several compiled packages had to be reinstalled after earlier Python-version conflicts.

---

## 7. Benchmark interpretation

Bonus NB6 was executed in mini mode and created:

- `data/eval/benchmark_results.json`
- `submission/screenshots/07-benchmark-comparison.png`

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | NaN | NaN | unavailable |
| GSM8K | NaN | NaN | unavailable |
| MMLU | NaN | NaN | unavailable |
| AlpacaEval-lite | NaN | NaN | unavailable |

The benchmark pipeline ran and saved the expected JSON/plot artifacts, but the local run did not produce valid numeric scores. `benchmark_results.json` contains `NaN` for all metrics, so I should not interpret the chart as evidence that DPO improved or degraded benchmark performance. The honest conclusion is that the benchmark harness wiring is present, but a full scored benchmark should be rerun in a cleaner Linux/Colab environment or after fixing the lm-eval output parsing on Windows.

This still matches the lesson from deck section 8.1: alignment evaluation is fragile, and missing/noisy benchmark numbers should not be overclaimed. For this local run, the stronger evidence is artifact completion across SFT, preference data, DPO adapter saving, and qualitative comparison generation.

---

## Bonus status

- [x] NB6 benchmark mini executed and artifacts saved
- [ ] Beta sweep not run
- [ ] Hugging Face Hub push not completed
- [ ] GGUF export / llama.cpp smoke test not completed
- [ ] W&B run not configured
- [ ] Cross-judge comparison pending API keys or manual rubric
- [ ] Pair work with: solo

GGUF export was left incomplete because `llama-cpp-python` installation on native Windows hit a Windows long-path issue in the vendored `llama.cpp` tree. Recommended next step for GGUF is WSL/Colab, or enabling Windows long paths before reinstalling `llama-cpp-python`.

---

## What surprised me most

The surprising part was not only GPU memory. The Windows environment details mattered just as much: missing `make`, path-length limits, cache location, optional transfer backends, and encoding issues all affected whether the notebooks could run end to end. The practical lesson is that a smaller reliable model with honest reporting is better than a larger target model that fails before producing artifacts.
