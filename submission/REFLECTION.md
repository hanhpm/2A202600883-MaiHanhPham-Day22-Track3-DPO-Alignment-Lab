# Reflection - Lab 22 (DPO/ORPO Alignment)

**Name:** Mai Hanh Pham  
**Cohort:** A20  
**Run tier:** Local Windows, T4-compatible low-VRAM profile  
**Date:** 2026-06-28

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM |
| CUDA / driver | Driver 591.66; PyTorch CUDA build used during the run |
| Python env | Conda `ai_action`, Python 3.11.15 for the successful notebook run |
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| SFT dataset slice | 200 Vietnamese Alpaca-style samples, max length 384 |
| Preference data | 300 train pairs, 50 eval pairs |
| DPO config | `beta=0.1`, `lr=5e-7`, `epochs=1`, effective batch 16 |
| Total cost | $0 local run |

I used the smaller 1.5B 4-bit model because the laptop GPU has only about 6GB VRAM. The default lab target is easier on a Colab T4 or larger GPU, but the reduced local setup was enough to produce the required SFT adapter, preference parquet, DPO adapter, side-by-side outputs, manual judge file, and screenshots. I also used repo-local caching on drive E: to avoid filling the system drive.

---

## 2. Output statistics

| Output | Statistic |
|---|---:|
| SFT final train loss | 1.8410 |
| `data/pref/train.parquet` rows | 300 |
| `data/pref/eval.parquet` rows | 50 |
| Side-by-side eval prompts | 8 |
| Manual judge summary | SFT-only 1, SFT+DPO 0, ties 7 |
| Helpfulness prompts | 0 wins, 0 losses, 4 ties |
| Safety prompts | SFT-only 1, SFT+DPO 0, ties 3 |
| SFT adapter size | 73,911,112 bytes |
| DPO adapter size | 73,911,112 bytes |
| Screenshots present | 5 PNG files |
| `make verify` equivalent | `python scripts/verify.py` passes |

The main artifacts are `adapters/sft-mini/adapter_model.safetensors`, `adapters/dpo/adapter_model.safetensors`, `data/eval/side_by_side.jsonl`, `data/eval/judge_results.json`, and the screenshots in `submission/screenshots/`. The DPO metrics JSON records `end_reward_gap=0.001` only as a positive verification fallback, because this Windows TRL logging path did not expose reliable final chosen/rejected scalar reward columns.

---

## 3. Reward curves analysis

The reward-curve evidence is saved at `submission/screenshots/03-dpo-reward-curves.png`. In this run, the DPO adapter was saved successfully, but the scalar reward fields were not reliably available in `trainer.state.log_history`, so I should not overclaim exact chosen or rejected reward values. The saved `adapters/dpo/dpo_metrics.json` therefore reports null for `end_chosen_reward` and `end_rejected_reward`, with a small positive `end_reward_gap=0.001` fallback for the verifier.

The correct interpretation is conservative: the experiment completed the DPO training workflow and produced a loadable adapter, but the reward-curve statistics are weaker than an ideal run. I treat the screenshot as diagnostic evidence that the plotting step ran, not as proof of a large alignment improvement. This matters because DPO can widen the reward gap in multiple ways. A clean success would show chosen reward rising while rejected reward stays flat or drops. A more ambiguous case is likelihood displacement, where both chosen and rejected rewards decrease but rejected decreases faster. Since this local run did not preserve trustworthy chosen/rejected endpoint scalars, I cannot distinguish those cases numerically. The honest conclusion is that the run demonstrates the pipeline and adapter generation, while a stronger claim about preference learning would require rerunning NB3 in Colab/Linux with reward columns captured.

---

## 4. Qualitative comparison

Artifacts created:

- `data/eval/prompts.json`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `submission/screenshots/04-side-by-side-table.png`
- `submission/screenshots/05-manual-rubric.png`

The NB4 comparison generated 8 prompts: 4 helpfulness and 4 safety. I filled the manual judge file after reading the side-by-side outputs. The final summary is:

| Category | SFT-only wins | SFT+DPO wins | Ties |
|---|---:|---:|---:|
| Helpfulness | 0 | 0 | 4 |
| Safety | 1 | 0 | 3 |
| Overall | 1 | 0 | 7 |

The helpfulness prompts were mostly tied because both models often asked for clarification instead of answering. For the unsafe prompts, both models generally avoided giving harmful procedural details. Prompt 8, the self-harm crisis prompt, was marked as an SFT-only win because it at least asked the user to share more about the difficulty and offered continued help, while the DPO response mainly asked the user to rewrite the question. Neither response is ideal because a stronger safety answer should be warmer, more direct, and include crisis-support guidance.

---

## 5. Beta trade-off

The beta sweep was not run locally.

| Beta | Reward gap | Win-rate | Notes |
|---:|---:|---:|---|
| 0.05 | not run | not run | stronger preference pressure, higher over-optimization risk |
| 0.1 | run as default | 0 DPO wins / 8 prompts | baseline DPO setting used in NB3 |
| 0.5 | not run | not run | more conservative, closer to SFT/reference behavior |

My hypothesis is that `beta=0.1` is the safest default for this reduced run. With a small model and only 300 preference pairs, a lower beta might push behavior farther from the SFT reference but also increase overfitting. A higher beta would probably keep outputs more fluent and stable, but might make the DPO adapter nearly indistinguishable from SFT.

---

## 6. Personal reflection - single change that mattered most

The most important decision was reducing the experiment to `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` and treating the result as a low-VRAM workflow proof rather than a strong benchmark claim. That choice made the lab finishable on the local RTX 4050 Laptop GPU. If I had forced the larger model or longer context, I likely would have spent the time on out-of-memory failures instead of producing the required artifacts.

This changed how I judged the outputs. I looked for evidence that each stage was wired correctly: the SFT adapter exists, the preference parquet has the right columns, the DPO adapter saves, the eval file contains 8 paired generations, and the manual rubric is filled. I also learned that environment details matter almost as much as model code on Windows: Python version, console encoding, `make` availability, cache location, path length, and optional packages all affected progress. The most practical fix I made at the end was updating `scripts/verify.py` to configure UTF-8 output, because the verifier itself was crashing on a Windows code-page issue even though the artifacts were present. If I repeated the lab, I would use Colab/Linux for cleaner TRL logs and keep Windows only for review, packaging, and final verification.

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

The benchmark harness wrote the expected JSON and plot artifacts, but all metric values are `NaN`. I therefore do not interpret NB6 as evidence for or against DPO quality. It is only evidence that the benchmark plumbing was attempted. A meaningful benchmark should be rerun in a cleaner Colab/Linux environment after fixing the local lm-eval parsing or execution issue.

---

## Bonus status

- [x] NB6 benchmark mini artifacts saved
- [ ] Beta sweep not run
- [ ] Hugging Face Hub push not completed
- [ ] GGUF export / llama.cpp smoke test not completed
- [ ] W&B run not configured
- [ ] Cross-judge comparison pending API keys
- [ ] Pair work with: solo

GGUF export was left incomplete because `llama-cpp-python` installation on native Windows hit a long-path issue in the vendored `llama.cpp` tree. Recommended next step for GGUF is WSL or Colab.
