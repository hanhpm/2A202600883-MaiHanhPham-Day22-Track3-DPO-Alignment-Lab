# Lab 22 Completion Plan

## Current Status

This repository is still a clean lab template. The core training and evaluation artifacts have not been generated yet.

Verified local environment:

- Requested env: `conda activate ai_action`
- Python: `3.14.4`
- Torch: CPU-only
- CUDA visible to PyTorch in `ai_action`: no
- Machine GPU from `nvidia-smi`: NVIDIA RTX 4050 Laptop GPU, 6GB VRAM
- Lab requirement for default T4 tier: about 12GB+ VRAM

Conclusion: do not run the full lab locally in `ai_action`. Use Colab T4 or another GPU with at least 12GB VRAM.

## Goal

Complete the graded core lab:

1. NB1: build SFT-mini adapter
2. NB2: prepare preference data
3. NB3: train DPO adapter
4. NB4: compare SFT-only vs SFT+DPO
5. Fill reflection
6. Add screenshots
7. Run verify
8. Submit public GitHub URL to LMS

Optional bonus:

1. NB5: merge + GGUF deploy
2. NB6: benchmark
3. beta-sweep
4. HuggingFace Hub push

## Recommended Execution Path

Use Colab T4 for the core path.

```bash
git clone <your-public-or-private-working-repo-url>
cd Day22-Track3-DPO-Alignment-Lab
bash setup-colab.sh
make smoke
make pipeline
```

If `make` is unavailable in Colab, run the notebooks in this order:

```text
notebooks/01_sft_mini.ipynb
notebooks/02_preference_data.ipynb
notebooks/03_dpo_train.ipynb
notebooks/04_compare_and_eval.ipynb
```

## Core Artifact Checklist

After NB1:

- `adapters/sft-mini/adapter_config.json`
- `submission/screenshots/02-sft-loss.png`
- Notebook output includes one SFT generation sample

After NB2:

- `data/pref/train.parquet`
- `data/pref/eval.parquet`
- Notebook output prints 3 inspected preference examples

After NB3:

- `adapters/dpo/adapter_config.json`
- `adapters/dpo/dpo_metrics.json`
- `submission/screenshots/03-dpo-reward-curves.png`
- Notebook output prints chosen reward, rejected reward, and reward gap

After NB4:

- `data/eval/prompts.json`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `submission/screenshots/04-side-by-side-table.png`
- Notebook output includes win/loss/tie summary

## Screenshot Checklist

Minimum useful screenshots:

1. `01-setup-gpu.png`: `nvidia-smi` or `torch.cuda.get_device_name()` output
2. `02-sft-loss.png`: SFT loss curve
3. `03-dpo-reward-curves.png`: chosen, rejected, and reward gap curves
4. `04-side-by-side-table.png`: 8 prompts x 2 models table
5. `05-judge-output.png` or `05-manual-rubric.png`: judge/manual verdicts

Optional bonus screenshots:

1. `06-gguf-smoke.png`: GGUF load + generated response
2. `07-benchmark-comparison.png`: benchmark comparison chart
3. `bonus-beta-sweep.png`: beta sweep plot

## Verify Command

On Windows PowerShell:

```powershell
$env:PYTHONUTF8='1'
python .\scripts\verify.py
```

On Colab/Linux:

```bash
make verify
```

Expected core pass requires:

- SFT adapter exists
- DPO adapter exists
- preference parquet exists
- side-by-side eval exists
- judge results exist
- reflection edited
- at least 3 screenshots

## Bonus Commands

Run only after core NB1-NB4 pass.

```bash
make deploy
make bench
make beta-sweep
```

Bonus artifacts:

- `gguf/*.gguf`
- `data/eval/deploy_meta.json`
- `data/eval/benchmark_results.json`
- `submission/screenshots/07-benchmark-comparison.png`
- `submission/screenshots/bonus-beta-sweep.png`

## Reflection Writing Guide

Fill `submission/REFLECTION.md` using real numbers from:

- `adapters/dpo/dpo_metrics.json`
- NB1 final training loss
- NB3 reward curve plot
- NB4 judge summary
- optional `data/eval/benchmark_results.json`

Do not invent metrics. If a bonus task was not run, say it was not run and write a hypothesis instead.

Important grading points:

- Section 3 must discuss chosen and rejected reward curves separately.
- Section 4 must include 8 examples and win/loss/tie summary.
- Section 5 can be a beta-sweep hypothesis if beta-sweep was not run.
- Section 6 must be at least 150 words.
- Section 7 is only needed for benchmark bonus.

## Final Submission Steps

```bash
git add -A
git commit -m "Lab 22 submission - Mai Hanh Pham"
git push -u origin main
```

Then submit the public GitHub repository URL to VinUni LMS.

Keep the repository public until grades are released.
