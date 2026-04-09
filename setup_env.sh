#!/usr/bin/env bash
# =============================================================================
# HPC 环境配置脚本 — Hard Negative Construction
# 用法：bash setup_env.sh [conda_env_name]
# 示例：bash setup_env.sh hard_neg
# =============================================================================
set -e

ENV_NAME=${1:-hard_neg}
PYTHON_VERSION=3.10

echo "=========================================="
echo " Hard Neg — HPC Environment Setup"
echo " Conda env: $ENV_NAME"
echo "=========================================="

# ── Step 1: 创建 conda 环境 ────────────────────────────────────────────────
echo "[1/5] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Step 2: 安装 PyTorch（自动匹配 CUDA 版本）────────────────────────────
echo "[2/5] Installing PyTorch ..."
# 自动检测 CUDA 版本
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    echo "  Detected CUDA: $CUDA_VER"
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  [WARN] CUDA < 11, falling back to CPU build"
        pip install torch
    fi
else
    echo "  [WARN] nvcc not found, installing CPU build (GPU won't be used)"
    pip install torch
fi

# ── Step 3: 安装其余依赖 ────────────────────────────────────────────────────
echo "[3/5] Installing project dependencies ..."
pip install -r requirements.txt

# ── Step 4: 安装 spaCy 英文模型 ────────────────────────────────────────────
echo "[4/5] Downloading spaCy en_core_web_sm ..."
python -m spacy download en_core_web_sm

# ── Step 5: 预下载 Qwen3 模型 ──────────────────────────────────────────────
echo "[5/5] Pre-downloading Qwen3-1.7B (requires internet access on this node)"
echo "  Model will be cached at ~/.cache/huggingface/"
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "Qwen/Qwen3-1.7B"
print(f"  Downloading tokenizer: {model_id}")
AutoTokenizer.from_pretrained(model_id)
print(f"  Downloading model weights: {model_id}")
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
print("  Done.")
EOF

# ── 验证 ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Verification"
echo "=========================================="
python - <<'EOF'
import torch, transformers, spacy, pandas, pyarrow
print(f"  torch       : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  transformers: {transformers.__version__}")
print(f"  spacy       : {spacy.__version__}")
print(f"  pandas      : {pandas.__version__}")
print(f"  pyarrow     : {pyarrow.__version__}")
nlp = spacy.load("en_core_web_sm")
print(f"  en_core_web_sm: OK")
EOF

echo ""
echo "=========================================="
echo " Setup complete!  conda activate $ENV_NAME"
echo "=========================================="
