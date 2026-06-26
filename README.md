# Day 22 â€” DPO/ORPO Alignment Lab (Track 3)

Lab cho **AICB-P2T3 Â· NgÃ y 22 Â· DPO/ORPO Alignment â€” From SFT to Preference Learning**.
Build SFT-mini checkpoint â†’ train DPO adapter â†’ compare SFT-only vs SFT+DPO â†’ merge + GGUF + serve.

> Lab 22 lÃ  **lab alignment Ä‘áº§u tiÃªn trong khoÃ¡** â€” báº¡n Ä‘i tá»« SFT (Lab 21) sang preference learning, Ä‘o helpfulness/safety báº±ng judge, vÃ  export model deployable. Output cÃ³ thá»ƒ lÃ  1 **DPO-aligned VN model open-source publishable Ä‘áº§u tiÃªn end-to-end cá»§a khoÃ¡** (xem deck Â§5).

---

## Hai tier â€” chá»n cÃ¡i phÃ¹ há»£p

| Tier | Compute | Base model | SFT slice | DPO slice | Time | Khi nÃ o dÃ¹ng |
|---|---|---|---|---|---|---|
| **T4/local low-VRAM (default)** | Free Colab T4 16 GB / laptop GPU >= 6 GB | `Qwen2.5-1.5B-Instruct-bnb-4bit` | 200 VN Alpaca fallback / configurable | 300 preference pairs fallback / configurable | local smoke first | RTX 4050/3060/4060 laptop or Colab T4 |
| **BigGPU (full)** | Colab Pro A100/L4 / Kaggle T4Ã—2 / cloud H100 | `Qwen2.5-7B-bnb-4bit` | 1k VN Alpaca | 5k UltraFeedback | ~25 min core (NB1-4) | ÄÃ£ cÃ³ cloud GPU, muá»‘n faithful vá»›i deck demo (3.2 â†’ 4.1 helpfulness, A100 timing) |

> Cáº£ hai tier dÃ¹ng **cÃ¹ng notebook source** â€” Ä‘á»•i giá»¯a T4 vÃ  BigGPU báº±ng cÃ¡ch sá»­a `COMPUTE_TIER` trong `.env` (hoáº·c Ä‘á»•i badge launch URL bÃªn dÆ°á»›i).

> **VRAM math quan trá»ng:** DPO cháº¥m má»—i cÃ¢u dÆ°á»›i *cáº£* policy vÃ  reference. Vá»›i PEFT/LoRA, TRL **khÃ´ng** náº¡p model thá»© 2 -- nÃ³ táº¯t adapter Ä‘á»ƒ láº¥y reference forward pass trÃªn cÃ¹ng base 4-bit. VRAM cao hÆ¡n SFT lÃ  do **2 forward pass + giá»¯ cáº£ chosen láº«n rejected** trong batch (~1.5-2x activation memory cá»§a SFT), *khÃ´ng* pháº£i vÃ¬ 2 báº£n weights. ÄÃ³ lÃ  lÃ½ do T4 tier dÃ¹ng 3B (khÃ´ng 7B) vÃ  BigGPU tier yÃªu cáº§u A100/L4.

---

## Quick Start â€” T4 (recommended)

**Option 1: Free Colab (zero install)**

[![Open T4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb)

Click â†’ Runtime â†’ Change runtime type â†’ **T4 GPU** â†’ Run all.

**Option 2: Local laptop (>= 6 GB VRAM, 1.5B tier)**

```bash
git clone https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
cd Day22-Track3-DPO-Alignment-Lab
bash setup-laptop.sh    # ~5 min â€” venv + deps + cuda probe + smoke test
make smoke              # import + GPU check (no training)
make pipeline           # CORE: sft â†’ data â†’ dpo â†’ eval (NB1-4, ~30 min)
make verify             # pre-submission gatekeeper
```

Yeu cau: **Python 3.10-3.12**, NVIDIA GPU >= 6 GB VRAM for the 1.5B tier, CUDA-capable PyTorch.

### Táº¥t cáº£ lá»‡nh `make`

```
make help            Show this help
make setup           Auto-detect Colab vs laptop, install deps + smoke check
make smoke           Import + GPU check (scripts/verify.py --smoke)
make sft             NB1 â€” build SFT-mini checkpoint (~10 min T4 / ~5 min A100)
make data            NB2 â€” preference data prep (~2 min)
make dpo             NB3 â€” DPO training (~15 min T4 / ~12 min A100)
make eval            NB4 â€” side-by-side comparison + optional API judge
make pipeline        CORE: run NB1 â†’ NB4 in order (~30 min T4)
make deploy          NB5 (OPTIONAL) â€” merge + GGUF + llama.cpp smoke
make bench           NB6 (OPTIONAL) â€” IFEval/GSM8K/MMLU + 4-bar plot (~30 min T4)
make pipeline-full   Core + optional NB5 + NB6
make beta-sweep      Bonus rigor: re-run NB3 with Î² âˆˆ {0.05, 0.1, 0.5}
make verify          scripts/verify.py â€” gatekeeper (core passes without NB5/NB6)
make clean           rm adapters/ data/pref/ gguf/ __pycache__
```

---

## Quick Start â€” BigGPU (full)

```bash
bash setup-laptop.sh                     # base install
pip install -r requirements-biggpu.txt   # adds vllm, flash-attn, deepspeed
echo 'COMPUTE_TIER=BIGGPU' > .env        # flip the tier flag
make pipeline                             # ~30 min on A100
```

Hoáº·c Colab Pro / Kaggle: open `colab/Lab22_DPO_BigGPU.ipynb` (badge link sáº½ resolve sau khi push lÃªn GitHub).

---

## Cáº¥u trÃºc & tiáº¿n trÃ¬nh

| Notebook | Skill | Slide deliverable | Pass whenâ€¦ |
|---|---|---|---|
| `01_sft_mini` | Re-build Lab 21 SFT checkpoint inline (Unsloth + LoRA r=16, 1k VN Alpaca, 1 epoch) | Bullet 1 â€” base SFT artifact | adapter saves; loss decreases monotonically |
| `02_preference_data` | Load `argilla/ultrafeedback-binarized-preferences-cleaned`, format `prompt/chosen/rejected`, save Parquet | Bullet 2 â€” preference data ready | parquet written; chosen â‰  rejected; 3 examples printed |
| `03_dpo_train` | TRL `DPOTrainer(beta=0.1, lr=5e-7)` on SFT model + frozen reference; plot reward curves | Bullet 3 â€” DPO training + reward curves | adapter saves; reward gap > 0; chosen reward â†‘ (or â†“ explained per deck Â§3.4) |
| `04_compare_and_eval` | 8 fixed prompts Ã— {SFT, SFT+DPO} side-by-side; optional GPT-4o/Claude judge | Bullet 4 â€” helpfulness comparison | table renders; â‰¥ 8 examples; win/loss/tie counts reported |
| `05_merge_deploy_gguf` **(OPTIONAL)** | `merge_and_unload()` â†’ GGUF Q4_K_M â†’ llama-cpp-python smoke test | Bullet 5 â€” deployable artifact | GGUF < 5 GB; smoke prompt returns coherent VN |
| `06_benchmark` **(OPTIONAL)** | IFEval + GSM8K + MMLU (sampled) + AlpacaEval-lite on SFT-only vs SFT+DPO; 4-bar comparison plot | Bullet 6 â€” quantitative benchmark | `benchmark_results.json` written; 4 deltas annotated in plot; Reflection Â§7 explains alignment-tax pattern |

**Source format:** Notebooks live as Jupytext `.py` files (small, easy to review). `setup-laptop.sh` and `make smoke` auto-convert to `.ipynb`. Edit `.ipynb` in Jupyter and Jupytext keeps both in sync.

**Colab variant:** `colab/Lab22_DPO_T4.ipynb` and `colab/Lab22_DPO_BigGPU.ipynb` are stitched-together single-file `.ipynb` â€” same content as the 5 Jupytext sources but ready to launch via badge. Pick the laptop path or the Colab path; both produce identical artifacts.

---

## Slide section â†’ notebook map

Tra ngÆ°á»£c tá»« slide báº¡n nhá»› trong lecture vá» cell trong notebook:

| Deck section | Slide topic | Notebook |
|---|---|---|
| Â§1 (Táº¡i sao SFT chÆ°a Ä‘á»§?) | Distribution shift, KL drift | `01_sft_mini.py` (má»¥c Ä‘Ã­ch) |
| Â§3.1 (DPO loss derivation) | Bradley-Terry â†’ log-ratio | `03_dpo_train.py` cell Â§3 |
| Â§3.2 (Î² tuning) | Trade-off conservative vs aggressive | `03_dpo_train.py` cell Â§5 (bonus Î²-sweep) |
| Â§3.4 (Failure modes) | Likelihood displacement, length hacking | `03_dpo_train.py` warning cell |
| Â§5.2 (TRL implementation) | `DPOConfig` hyperparameters | `03_dpo_train.py` cell Â§2 |
| Â§5.4 (VN landscape) | VinaLLaMA / PhoGPT / Vistral / SeaLLM | `02_preference_data.py` callout + `BONUS-CHALLENGE.md` provocation 1 |
| Â§8.1â€“Â§8.5 (ÄÃ¡nh giÃ¡ Alignment) | Static / Judge / Reward-Model / VN landscape | `06_benchmark.py` |
| Â§9.1 (Demo) | UltraFeedback 2k, 30 min A100, 3.2 â†’ 4.1 | `04_compare_and_eval.py` |
| Â§9.2b (Tulu 3 stats) | +1.7 MATH / +3.3 GSM8K / +1.3 IFEval | reference numbers + `06_benchmark.py` measures *your* equivalents |

---

## Deliverable (6 notebook Ä‘Ã£ cháº¡y + áº£nh chá»¥p + reflection)

Mapping 1-to-1 vá»›i slide deliverable bullets:

1. **NB1** â€” `adapters/sft-mini/` written; `01_sft_loss.png` shows monotonic decrease.
2. **NB2** â€” `data/pref/train.parquet` with prompt/chosen/rejected columns; 3 inspected examples printed.
3. **NB3** â€” `adapters/dpo/` written; reward gap plot saved as `03_dpo_reward_curves.png`.
4. **NB4** â€” `04_side_by_side_table.png` + win/loss/tie summary (8 prompts Ã— 2 models).
5. **NB5 (OPTIONAL/bonus)** â€” `gguf/lab22-dpo-Q4_K_M.gguf` exists; `06_gguf_smoke.png` shows llama.cpp output.
6. **NB6 (OPTIONAL/bonus)** â€” `data/eval/benchmark_results.json` + `07-benchmark-comparison.png` 4-bar chart with deltas annotated; REFLECTION Â§7 interprets alignment-tax pattern (deck Â§8.1).

Cháº¥m Ä‘iá»ƒm: xem [`rubric.md`](rubric.md). **Tá»•ng 100 pts â†’ Track-3 Daily Lab (30%)** + 20 pts bonus rigor add-ons (Î²-sweep, HF push, W&B, GGUF release).

---

## Tech stack

| Layer | Tool | Version | Why |
|---|---|---|---|
| **Training** | Unsloth | â‰¥ 2025.10 | Patched kernels, 7B-on-T4 viable, matches Day 21 reference |
| **Trainers** | TRL | â‰¥ 0.12, < 0.20 | `DPOTrainer` + `DPOConfig` (deck Â§5.2 surface) |
| **Adapters** | PEFT | â‰¥ 0.13 | LoRA r=16 Î±=32; reference model loaded as frozen 4-bit |
| **Quantization** | bitsandbytes | â‰¥ 0.44 | NF4 base + bf16 LoRA |
| **Data** | datasets + pyarrow | â‰¥ 3.1 | UltraFeedback + VN Alpaca slices |
| **Local serving** | llama-cpp-python | â‰¥ 0.3 | GGUF Q4_K_M smoke test (CPU/Metal/CUDA) |
| **Cloud serving** | vllm (BigGPU only) | â‰¥ 0.6.4 | OpenAI-compat for production-style serve test |
| **Plotting** | matplotlib + pandas | â‰¥ 3.9 | Reward curves + side-by-side tables |

**Why not vLLM by default?** vLLM needs CUDA GPU + â‰¥ 16 GB VRAM and adds 3-5 min Docker/CUDA-toolkit install. For T4 tier we use llama-cpp-python which compiles inline in the wheel and works on CPU/Metal/CUDA. BigGPU tier gets vLLM as a final cell (informational on T4).

---

## Vibe-coding tips

Lab nÃ y thiáº¿t káº¿ cho **vibe-coding era**: báº¡n dÃ¹ng AI assistant trong terminal (Claude Code, Codex CLI, OpenCode) Ä‘á»ƒ generate boilerplate, focus vÃ o *judgment decisions* â€” chá»n dataset, chá»n Î², Ä‘á»c reward curve, judge output. Äá»c [`VIBE-CODING.md`](VIBE-CODING.md) **trÆ°á»›c khi báº¯t Ä‘áº§u NB1** (5â€“10 phÃºt) â€” file Ä‘Ã³ lÃ  general primer cover:

- Spec-Driven Development (SDD) vÃ  TDD trong LLM era
- Khi nÃ o delegate cho AI, khi nÃ o tá»± nghÄ©
- 5 prompt patterns DPO-specific (diagnose chosen reward drop, generate VN prompts, critique config, translate UltraFeedback, judge outputs)
- CLI tool recommendations (Claude Code / Codex CLI / OpenCode)
- 3 anti-patterns phá»• biáº¿n trong alignment work

Má»—i notebook cÅ©ng cÃ³ **vibe-coding callout** á»Ÿ cuá»‘i: nÃ³i rÃµ subtask nÃ o *nÃªn* delegate cho AI, subtask nÃ o *pháº£i* báº¡n tá»± nghÄ© (hint: reward curve interpretation vÃ  Î² chá»n = think-hard zone).

---

## Bonus Challenge â€” Build something real (optional, ungraded)

Má»™t sÃ¢n chÆ¡i **khÃ´ng cÃ³ Ä‘iá»ƒm sá»‘** â€” khÃ´ng deadline, khÃ´ng rubric. Má»¥c Ä‘Ã­ch: cho báº¡n Ä‘em **domain knowledge cÃ¡ nhÃ¢n** vÃ o 1 model align tháº­t, ship nhÆ° sáº£n pháº©m cho 1 audience cá»¥ thá»ƒ. Má»—i provocation há»i báº¡n 4 cÃ¢u: *Ai dÃ¹ng?* â€” *Báº¡n Ä‘em domain gÃ¬ vÃ o?* â€” *Model lÃ m gÃ¬ cho há»?* â€” *Output ship nhÆ° tháº¿ nÃ o?*

Äá» xuáº¥t 5 provocations sáºµn â€” báº¡n pick 1 hoáº·c invent your own:

1. **Subject tutor** cho mÃ´n báº¡n Ä‘ang há»c (toÃ¡n, hoÃ¡, sá»­, láº­p trÃ¬nh...) â€” scaffolded pedagogy, khÃ´ng pháº£i Ä‘Ã¡p Ã¡n
2. **Customer-service chatbot** cho 1 doanh nghiá»‡p Viá»‡t cá»¥ thá»ƒ (cafe, shop, sá»­a xe...) â€” on-brand, cÃ³ CTA
3. **Job-shadow assistant** cho 1 nghá» báº¡n quan sÃ¡t (Grab driver, shipper, lá»… tÃ¢n...) â€” ngáº¯n, action-oriented
4. **Domain-safe assistant** cho 1 lÄ©nh vá»±c nháº¡y cáº£m (sá»©c khoáº» tinh tháº§n, phÃ¡p lÃ½, tÃ i chÃ­nh...) â€” cÃ³ boundary rÃµ + hotline VN
5. **Style mimic** â€” model viáº¿t kiá»ƒu 1 ngÆ°á»i/tá»• chá»©c báº¡n admire

Full provocations: [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md) (tiáº¿ng Viá»‡t) Â· [`BONUS-CHALLENGE-EN.md`](BONUS-CHALLENGE-EN.md) (English). Format: brainstorm-first, code-second, lÃ m Ä‘Ã´i/triple OK. Output: 1 portfolio piece cÃ³ thá»ƒ chá»‰ vÃ o nÃ³i "tÃ´i build cÃ¡i nÃ y, audience X, dÃ¹ng Ä‘á»ƒ Y."

> Bonus **khÃ´ng** áº£nh hÆ°á»Ÿng core grade. Pháº§n thÆ°á»Ÿng thá»±c sá»± lÃ  1 portfolio piece phá»¥c vá»¥ *ai Ä‘Ã³ cá»¥ thá»ƒ* + feedback báº±ng vÄƒn báº£n tá»« giáº£ng viÃªn vá» *application thinking* cá»§a báº¡n.

---

## Cáº¥u trÃºc repo

```
.
â”œâ”€â”€ README.md                       # báº¡n Ä‘ang Ä‘á»c
â”œâ”€â”€ HARDWARE-GUIDE.md               # T4 vs BigGPU decision tree
â”œâ”€â”€ VIBE-CODING.md                  # vibe-coding workflow tips (5-10 phÃºt Ä‘á»c)
â”œâ”€â”€ BONUS-CHALLENGE.md              # creative sandbox brief (tiáº¿ng Viá»‡t)
â”œâ”€â”€ BONUS-CHALLENGE-EN.md           # creative sandbox brief (English)
â”œâ”€â”€ rubric.md                       # 100-pt grading + 20 pt bonus rigor add-ons
â”œâ”€â”€ Makefile                        # tier-aware orchestration
â”œâ”€â”€ setup-colab.sh                  # one-line Colab install
â”œâ”€â”€ setup-laptop.sh                 # local venv + cuda probe
â”œâ”€â”€ requirements.txt                # T4 baseline deps
â”œâ”€â”€ requirements-biggpu.txt         # BigGPU extras (vllm, flash-attn)
â”œâ”€â”€ pyproject.toml                  # for `uv` users
â”œâ”€â”€ .env.example                    # env template (COMPUTE_TIER, API keys)
â”œâ”€â”€ notebooks/                      # 6 Jupytext .py files (source of truth)
â”‚   â”œâ”€â”€ 01_sft_mini.py              # build SFT checkpoint inline
â”‚   â”œâ”€â”€ 02_preference_data.py       # load + format UltraFeedback
â”‚   â”œâ”€â”€ 03_dpo_train.py             # TRL DPOTrainer + reward curves
â”‚   â”œâ”€â”€ 04_compare_and_eval.py      # SFT-only vs SFT+DPO + judge
â”‚   â”œâ”€â”€ 05_merge_deploy_gguf.py     # merge + GGUF + llama.cpp smoke
â”‚   â””â”€â”€ 06_benchmark.py             # IFEval/GSM8K/MMLU/AlpacaEval-lite + 4-bar plot
â”œâ”€â”€ colab/                          # Colab-launchable .ipynb mirrors
â”‚   â”œâ”€â”€ Lab22_DPO_T4.ipynb
â”‚   â””â”€â”€ Lab22_DPO_BigGPU.ipynb
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ prepare_preference_data.py  # CLI wrapper for NB2 logic
â”‚   â”œâ”€â”€ train_dpo.py                # CLI wrapper for NB3 logic
â”‚   â”œâ”€â”€ eval_judge.py               # OpenAI/Anthropic judge â€” falls back to manual
â”‚   â”œâ”€â”€ merge_and_gguf.py           # CLI wrapper for NB5 logic
â”‚   â””â”€â”€ verify.py                   # pre-submission gatekeeper
â”œâ”€â”€ data/                           # gitignored; populated by NB2 / scripts
â”œâ”€â”€ adapters/                       # gitignored; SFT + DPO outputs
â”œâ”€â”€ submission/
â”‚   â”œâ”€â”€ REFLECTION.md               # personal report template (6 sections)
â”‚   â””â”€â”€ screenshots/                # add 6 required + 3 optional screenshots
â””â”€â”€ solutions/                      # released after submission deadline
    â””â”€â”€ README.md
```

---

## Common gotchas

| Triá»‡u chá»©ng | Fix |
|---|---|
| OOM ngay khi load model | Dung local low-VRAM path: `Qwen2.5-1.5B-Instruct-bnb-4bit`, `MAX_LEN=384`, restart runtime, then rerun. |
| `chosen_rewards` khÃ´ng tÄƒng | BÃ¬nh thÆ°á»ng á»Ÿ 100 step Ä‘áº§u. Sau 500 step náº¿u váº«n flat â†’ giáº£m `beta` 0.1 â†’ 0.05 hoáº·c tÄƒng `lr` 5e-7 â†’ 1e-6 |
| `chosen_rewards` *giáº£m* mÃ  reward gap *tÄƒng* | ÄÃ³ lÃ  **likelihood displacement** (deck Â§3.4). BÃ¬nh thÆ°á»ng á»Ÿ DPO; ghi vÃ o REFLECTION Â§ "Î² trade-off" |
| `RuntimeError: padding token is not set` | Add `tokenizer.pad_token = tokenizer.eos_token` trÆ°á»›c khi táº¡o trainer |
| Unsloth + TRL version mismatch | Pin: `unsloth>=2025.10 trl>=0.12,<0.20`. Náº¿u lá»—i sau Unsloth update, downgrade Unsloth |
| GGUF merge fails vá»›i "tied weights" | XoÃ¡ `model.config.tie_word_embeddings` trÆ°á»›c `merge_and_unload()` |
| Colab T4 OOM at DPO step 1 | TÄƒng `gradient_accumulation_steps` 8 â†’ 16, giáº£m `per_device_train_batch_size` 1 â†’ 1 (already min), giáº£m `max_length` 512 â†’ 384 |
| llama-cpp-python wheel install fails on Windows | Enable Windows long paths, install it separately, or run NB5 in WSL/Colab. CUDA example: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`. |
| `lm_eval` import fails | `pip install "lm-eval[ifeval,math]>=0.4.5"` â€” extras pull `langdetect` and `sympy` |
| NB6 IFEval crashes with "DataLoader" worker | lm-eval 0.4.x compat â€” set env `HF_DATASETS_TRUST_REMOTE_CODE=1` and rerun |
| NB6 GSM8K accuracy = 0.000 | Few-shot prompts not loading. Verify `--num_fewshot 8` reaches the harness; downgrade lm-eval if 0.4.6 ships changes |
| NB6 takes > 90 min on T4 | Lower `LIMIT_MMLU` (default 500) and `LIMIT_GSM8K` (default 500) further. Bench tier checks env first. |

---

## Submission

**KHÃ”NG cáº§n PR â€” chá»‰ submit GitHub URL cÃ´ng khai vÃ o VinUni LMS.**

1. **Fork hoáº·c copy repo nÃ y lÃªn GitHub account cá»§a báº¡n**, set repo **public**.
   ```bash
   git init -b main
   git remote add origin https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
   ```
2. HoÃ n thÃ nh 5 notebooks (giá»¯ output cells trong `.ipynb`).
3. Add áº£nh chá»¥p vÃ o `submission/screenshots/` (xem [`submission/screenshots/README.md`](submission/screenshots/README.md) Ä‘á»ƒ biáº¿t list 6+3).
4. Äiá»n [`submission/REFLECTION.md`](submission/REFLECTION.md) (6 sections, â‰¥150 tá»« Â§3 + Â§6).
5. `make verify` â€” pre-submission gatekeeper. Náº¿u fail, fix vÃ  rerun.
6. Push lÃªn public repo:
   ```bash
   git add -A
   git commit -m "Lab 22 submission â€” <Há» TÃªn>"
   git push -u origin main
   ```
7. **Paste public GitHub URL cá»§a báº¡n vÃ o Ã´ submission cá»§a Day 22 trong VinUni LMS.** KhÃ´ng cáº§n PR. KhÃ´ng cáº§n fork-back.

> **Quan trá»ng:** Repo pháº£i **public** Ä‘áº¿n khi Ä‘iá»ƒm Ä‘Æ°á»£c cÃ´ng bá»‘. Náº¿u private, grader khÃ´ng xem Ä‘Æ°á»£c â†’ 0 Ä‘iá»ƒm.

**Submission Options A / B / C** (cÃ¹ng convention vá»›i Day 21):
- **A â€” Lightweight ZIP** (default): GitHub repo + executed notebooks + screenshots + REFLECTION
- **B â€” Professional** (+5 bonus): A + adapters pushed to HuggingFace Hub via `huggingface-cli upload`
- **C â€” Code-only**: Repo + report, khÃ´ng weights (cho há»c viÃªn háº¿t storage Colab)

---

## Acknowledgments

- **Slide deck:** [`day22/day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex`](../day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex)
- **Sibling Day 21 lab** (LoRA/QLoRA fine-tuning, the SFT predecessor): [VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA)
- **Stack:** Unsloth (Daniel Han + Mike Han), TRL (Hugging Face), PEFT, bitsandbytes, llama.cpp
- **Datasets:** UltraFeedback (Argilla), `5CD-AI/Vietnamese-alpaca-cleaned`

---

Â© VinUniversity AICB program Â· A20 cohort 2026 Â· Track 3 Day 22.


