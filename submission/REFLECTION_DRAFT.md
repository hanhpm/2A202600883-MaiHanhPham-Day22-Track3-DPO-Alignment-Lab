# Reflection Draft - Lab 22 DPO/ORPO Alignment

Use this draft after the notebooks have been executed. Replace every `TODO` with real numbers or observations from your run.

## 1. Setup

| Item | Value |
|---|---|
| GPU | TODO: e.g. Free Colab T4 16GB |
| CUDA / driver | TODO |
| Base model | TODO: `unsloth/Qwen2.5-3B-bnb-4bit` for T4 |
| SFT dataset slice | TODO: usually 1000 samples |
| Preference dataset slice | TODO: usually 1000 or 2000 pairs depending notebook/env |
| `COMPUTE_TIER` env | TODO: T4 or BIGGPU |
| Total cost | TODO |

## 2. DPO Experiment Results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time | TODO | TODO |
| VRAM peak | TODO | TODO |
| Final loss | TODO | TODO |
| Reward gap chosen - rejected | n/a | TODO |
| Mean output length | TODO | TODO |

## 3. Reward Curves Analysis

TODO: Write at least 100 words after viewing `03-dpo-reward-curves.png`.

Suggested structure:

The DPO reward curves show that the chosen reward [increased/decreased/stayed flat] while the rejected reward [increased/decreased/stayed flat]. The final reward gap was TODO, which means the model learned to separate preferred and rejected responses by TODO. The most important detail is whether the gap grew because chosen responses became more likely, or because rejected responses became less likely faster. In my run, TODO. This matters because DPO can produce likelihood displacement: the gap improves while both chosen and rejected rewards go down. If that happened, the result is still a useful alignment signal, but it should be interpreted carefully rather than treated as a simple helpfulness win.

## 4. Qualitative Comparison

| # | Prompt category | Winner | Note |
|---|---|---|---|
| 1 | helpfulness | TODO | TODO |
| 2 | helpfulness | TODO | TODO |
| 3 | helpfulness | TODO | TODO |
| 4 | helpfulness | TODO | TODO |
| 5 | safety | TODO | TODO |
| 6 | safety | TODO | TODO |
| 7 | safety | TODO | TODO |
| 8 | safety | TODO | TODO |

Win/loss/tie summary: TODO

Judge used: TODO: manual rubric / gpt-4o-mini / claude-haiku

## 5. Beta Trade-Off

If beta-sweep was not run:

I did not run the beta-sweep bonus. My hypothesis is that a smaller beta such as 0.05 would allow the policy to move more aggressively away from the reference model, likely increasing the reward gap but also increasing the risk of over-optimization or shorter, less natural answers. A larger beta such as 0.5 should keep the model closer to the SFT reference, which may preserve fluency and reduce regressions but produce a weaker preference-learning effect. I would expect the default beta 0.1 to be the best compromise for this small lab-scale UltraFeedback slice.

## 6. Personal Reflection

TODO: Write at least 150 words.

Suggested draft:

The decision that mattered most in this lab was choosing the compute tier. The alternative was to force the run on my local Windows laptop, but the local setup had an RTX 4050 Laptop GPU with only 6GB VRAM, while the T4 tier expects roughly 12GB or more for the 3B DPO path. I chose to treat Colab T4 as the correct execution environment instead of spending time fighting local OOM and package mismatch issues. This choice matters because DPO is much more memory-intensive than SFT: it compares chosen and rejected completions and needs both policy and reference behavior during training. If I redid the lab tomorrow, I would start directly on Colab, capture the GPU screenshot first, run the core pipeline before touching any bonus tasks, and only attempt GGUF or benchmark after `make verify` passes. The main lesson is that alignment experiments are not only about model code; the compute plan is part of the experimental design.

## 7. Benchmark Interpretation

Only fill this section if NB6 was run.

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | TODO | TODO | TODO |
| GSM8K | TODO | TODO | TODO |
| MMLU | TODO | TODO | TODO |
| AlpacaEval-lite | TODO | TODO | TODO |

TODO: Write at least 150 words if benchmark results exist.

## Bonus

- [ ] beta-sweep
- [ ] HuggingFace Hub push
- [ ] GGUF release
- [ ] W&B run link
- [ ] cross-judge comparison
- [ ] creative bonus challenge
