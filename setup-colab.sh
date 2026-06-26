#!/usr/bin/env bash
# Colab one-line setup. Drops the venv layer (Colab is already Python-isolated)
# and skips the CUDA probe (Colab notebooks already display GPU info in their
# UI). Just installs deps + auto-tier-detects + converts notebooks.
#
# Usage in Colab cell:
#     !git clone https://github.com/<user>/Day22-Track3-DPO-Alignment-Lab.git
#     %cd Day22-Track3-DPO-Alignment-Lab
#     !bash setup-colab.sh

set -euo pipefail

echo "[colab] Day 22 lab â€” Colab setup"
echo "[colab] Stack: unsloth + trl + peft + bitsandbytes + llama-cpp-python"
echo

# â”€â”€ 1. Auto-detect tier from torch.cuda â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TIER=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("CPU")
else:
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("BIGGPU" if gb >= 22 else "T4")
PY
)
echo "[colab] Detected tier: $TIER"

case "$TIER" in
  CPU)
    echo "[colab] No GPU detected. Runtime â†’ Change runtime type â†’ T4 GPU, then retry."
    exit 1
    ;;
  T4)
    echo "[colab] T4 (or similar 16 GB) tier â€” using Qwen2.5-3B"
    ;;
  BIGGPU)
    echo "[colab] BigGPU (A100 / L4) tier â€” using Qwen2.5-7B"
    ;;
esac

# â”€â”€ 2. Install deps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Colab pre-installs torch. Keep notebook/kernel packages stable and avoid the
# transformers 4.57 -> torchcodec import path that currently breaks Unsloth on
# Colab's torch 2.10/cu128 image.
pip install -q -r requirements.txt
pip uninstall -y -q torchcodec || true
pip install -q "transformers>=4.46,<4.57"

# Keep Colab's own ipykernel/jupyter-server versions intact; we only need
# jupytext + nbconvert for Makefile notebook execution.
pip install -q -U "jupytext>=1.16,<2.0" "nbconvert>=7,<8"
if [ "$TIER" = "BIGGPU" ]; then
  echo "[colab] Installing BigGPU extras (vllm, flash-attn) â€” may take 3-5 min"
  pip install -q -r requirements-biggpu.txt || echo "[colab] WARNING: vllm/flash-attn install failed; vLLM cell in NB5 will skip"
fi

# â”€â”€ 3. Convert Jupytext sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
jupytext --to notebook --update notebooks/*.py 2>/dev/null || jupytext --to notebook notebooks/*.py

# â”€â”€ 4. .env scaffold (optional in Colab) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
[ -f .env ] || cp .env.example .env
# Patch COMPUTE_TIER in .env to match auto-detect
sed -i.bak "s/^COMPUTE_TIER=.*/COMPUTE_TIER=$TIER/" .env && rm .env.bak

# â”€â”€ 5. Make output folders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
mkdir -p data/pref adapters/sft-mini adapters/dpo gguf

cat <<EOF

[colab] Done â€” tier = $TIER.

In Colab, you can now either:

    1. Open notebooks/01_sft_mini.py â€” Jupytext will convert on first edit
    2. Or run the stitched single-file Colab notebook:
         - T4 tier:     colab/Lab22_DPO_T4.ipynb
         - BigGPU tier: colab/Lab22_DPO_BigGPU.ipynb
    3. Or use the make targets:

         !make smoke           # quick verification
         !make pipeline        # full run (~45 min T4 / ~30 min A100)

Tip: read VIBE-CODING.md before starting NB1.

EOF

