# HPC 端到端操作指引

从零开始在 HPC 集群上完成：数据构建 → 两阶段训练 → MTEB 评估的完整流程手册。

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [Step 1：登录节点 — 克隆代码](#2-step-1登录节点--克隆代码)
3. [Step 2：环境配置](#3-step-2环境配置)
4. [Step 3：数据准备](#4-step-3数据准备)
5. [Step 4：预下载模型权重](#5-step-4预下载模型权重)
6. [Step 5：数据构建作业](#6-step-5数据构建作业)
7. [Step 6：训练作业](#7-step-6训练作业)
   - [6a — Debug / Test 冒烟验证](#6a--debug--test-冒烟验证)
   - [6b — 正式全量训练](#6b--正式全量训练)
   - [6c — 拆分训练（时限受限）](#6c--拆分训练时限受限)
8. [Step 7：评估作业](#8-step-7评估作业)
9. [Step 8：监控与进度追踪](#9-step-8监控与进度追踪)
10. [Step 9：下载结果到本地](#10-step-9下载结果到本地)
11. [常见报错排查](#11-常见报错排查)

---

## 1. 整体流程概览

```
登录节点
  │
  ├── git clone / git pull          ← 拉取最新代码（含最新 commit）
  ├── bash setup_env.sh hard_neg    ← 创建 conda 环境 + 安装依赖
  ├── 上传 / 下载 原始 parquet      ← 数据文件不入 git
  ├── 预下载模型权重                ← 计算节点无公网，必须提前下载
  │     ├── Qwen/Qwen3-1.7B        （LLM 路径构建）
  │     └── all-MiniLM-L6-v2       （训练基础模型 + MTEB 基准）
  │
  └── SLURM 作业
        │
        ├── [构建] run_construct.sh   ~5-24h（含 Regular + LLM）
        │
        ├── [训练] run_train_test.sh  ~30min  ← 先跑这个验证流程
        ├── [训练] run_train_full.sh  ~6-8h A100
        │
        └── [评估] run_eval.sh        ~1-2h per model
```

**实验设计（训练阶段）**

| 实验组 | 数据来源 | Phase 1 模型 | Phase 2 模型（×4 Loss） |
|--------|---------|-------------|------------------------|
| `llm`      | LLM 路径构建数据    | 1 个 checkpoint | 4 个模型 |
| `regular`  | Regular 路径构建数据 | 1 个 checkpoint | 4 个模型 |
| `combined` | LLM + Regular 合并  | 1 个 checkpoint | 4 个模型 |

Phase 2 的 4 种 Loss：`triplet_cascade`、`batch_hard`、`batch_semi_hard`、`batch_hard_soft_margin`

每种 Loss 从**同一 Phase 1 checkpoint** 独立加载，保证对比公平。

**预期输出（models/{RUN_ID}_compare/）**：

```
models/{RUN_ID}_compare/
├── logs/                          ← 全局日志 + 汇总 summary_all.json
├── filtered_data/                 ← T/T* 过滤结果（3 个实验各一个 JSON）
├── llm/
│   ├── phase1/final/              ← Phase 1 checkpoint
│   ├── phase2_triplet_cascade/final/
│   ├── phase2_batch_hard/final/
│   ├── phase2_batch_semi_hard/final/
│   └── phase2_batch_hard_soft_margin/final/
├── regular/   （同上结构）
└── combined/  （同上结构）
```

---

## 2. Step 1：登录节点 — 克隆代码

```bash
ssh your_username@your_hpc_host
cd /path/to/your/workspace

# 首次克隆
git clone https://github.com/Fu-fangteng/hard_negative_constuction.git hard_neg
cd hard_neg

# 已有旧版本则拉取最新
# cd hard_neg && git pull origin main
```

确认当前是最新提交：

```bash
git log --oneline -3
# 应能看到：da47b4f feat(train): implement phase2 training with T/T* hard-neg filter
```

---

## 3. Step 2：环境配置

```bash
# 在登录节点执行
bash setup_env.sh hard_neg
conda activate hard_neg
```

**验证环境**（关键依赖版本检查）：

```bash
python - <<'EOF'
import torch, transformers, spacy, sentence_transformers, mteb, matplotlib
print(f"torch               : {torch.__version__}")
print(f"CUDA available      : {torch.cuda.is_available()}")
print(f"transformers        : {transformers.__version__}")
print(f"sentence-transformers: {sentence_transformers.__version__}")
print(f"mteb                : {mteb.__version__}")
print(f"matplotlib          : {matplotlib.__version__}")
nlp = spacy.load("en_core_web_sm")
print("en_core_web_sm      : OK")
EOF
```

期望：
- `sentence-transformers >= 5.1.2`（必须，低版本缺少 BatchHardTripletLoss 等 API）
- `mteb >= 1.39.7`
- `CUDA available : True`（训练必须，构建 LLM 路径必须）

如果版本不满足，手动升级：

```bash
pip install "sentence-transformers>=5.1.2" "mteb>=1.39.7"
```

---

## 4. Step 3：数据准备

原始数据文件不入 git，需手动上传或在集群下载。

### 方式一：从本地 scp 上传

```bash
# 在本地终端执行
scp /path/to/hard_neg/data/raw/train-00000-of-00001.parquet \
    your_username@your_hpc_host:/path/to/workspace/hard_neg/data/raw/
```

### 方式二：集群登录节点直接下载（若有公网）

```bash
mkdir -p data/raw
conda activate hard_neg
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("sentence-transformers/nli-for-simcse", split="train")
ds.to_parquet("data/raw/train-00000-of-00001.parquet")
print(f"Downloaded {len(ds)} rows")
EOF
```

验证：

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/raw/train-00000-of-00001.parquet")
print(f"总行数: {len(df):,}")       # 期望 274,951
print(f"列名  : {df.columns.tolist()}")  # ['anchor', 'positive', 'negative']
EOF
```

如果**已经有构建好的数据**（`data/stage2/processed/` 已存在），可跳过数据构建步骤（Step 5），直接进入 Step 6 训练。

---

## 5. Step 4：预下载模型权重

计算节点通常无法访问互联网，**必须在登录节点提前下载**所有需要的模型。

```bash
conda activate hard_neg
export HF_HOME=/path/to/shared/.cache/huggingface   # ← 修改为共享存储路径
```

### 4a — all-MiniLM-L6-v2（训练基础模型，必须下载）

```bash
python - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("all-MiniLM-L6-v2 cached OK, embedding dim:", model.get_sentence_embedding_dimension())
EOF
```

### 4b — Qwen3-1.7B（LLM 路径数据构建，仅需 LLM 路径时下载）

```bash
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="auto")
print("Qwen3-1.7B cached OK")
EOF
```

验证两个模型都可以无网络加载：

```bash
python - <<'EOF'
import os; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
m = SentenceTransformer("all-MiniLM-L6-v2")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
print("Offline load OK")
EOF
```

---

## 6. Step 5：数据构建作业

> **如果已有 `data/stage2/processed/` 构建结果，可完全跳过此步骤。**

创建作业目录：

```bash
mkdir -p jobs logs
```

### 方案 A — Regular + LLM 一次跑完

```bash
cat > jobs/run_construct.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_construct
#SBATCH --partition=gpu              # ← 修改为你的 GPU 分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/construct_%j.out
#SBATCH --error=logs/construct_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg        # ← 修改
HF_CACHE=/path/to/shared/.cache/huggingface  # ← 修改
CONDA_ENV=hard_neg

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "$GPU_MEM" -ge 70000 ]; then BATCH=16
elif [ "$GPU_MEM" -ge 30000 ]; then BATCH=12
elif [ "$GPU_MEM" -ge 20000 ]; then BATCH=8
else BATCH=6; fi
echo "GPU mem: ${GPU_MEM}MiB  → llm_batch_size=${BATCH}"

python stage2/run_stage2.py \
    --input_path     data/raw/train-00000-of-00001.parquet \
    --output_base    data/stage2 \
    --methods        direct_negation_attack \
    --recognizer     both \
    --llm_model      Qwen/Qwen3-1.7B \
    --llm_batch_size "$BATCH"

echo "Done: $(date)"
JOBEOF

sbatch jobs/run_construct.sh
```

---

## 7. Step 6：训练作业

训练作业分三个层级：debug（冒烟）→ test（流程验证）→ full（正式）。

**先在 test 模式下确认整个流程没有错误，再提交正式作业。**

### 6a — Debug / Test 冒烟验证

先提交一个 test 作业，验证代码在真实 GPU 环境下跑通：

```bash
cat > jobs/run_train_test.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_train_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg        # ← 修改
HF_CACHE=/path/to/shared/.cache/huggingface  # ← 修改
CONDA_ENV=hard_neg

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1

echo "=== TRAIN TEST MODE ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python train_hard_neg.py --test

echo "=== TEST DONE: $(date) ==="
echo "Check models/ directory for output"
ls -lh models/ | tail -5
JOBEOF

sbatch jobs/run_train_test.sh
```

test 模式每组数据截断到 500 条，整个流程约 **20–30 分钟**。
输出目录：`models/{RUN_ID}_test_compare/`

完成后检查日志末尾的汇总表：

```bash
tail -30 logs/train_test_<job_id>.out
# 应能看到：
# Exp          Loss                         Train  P1-Orig  P2-Orig   ΔOrig  P1-Hard  P2-Hard   ΔHard
# llm          triplet_cascade               ...
# llm          batch_hard                    ...
# ...
```

### 6b — 正式全量训练

确认 test 跑通后，提交正式作业：

```bash
cat > jobs/run_train_full.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg        # ← 修改
HF_CACHE=/path/to/shared/.cache/huggingface  # ← 修改
CONDA_ENV=hard_neg

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1

echo "=== FULL TRAINING ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
nvidia-smi

python train_hard_neg.py

echo "=== TRAINING DONE: $(date) ==="
echo ""
echo "=== Output directory ==="
RUN_DIR=$(ls -td models/*_compare | grep -v debug | grep -v test | head -1)
echo "$RUN_DIR"
ls -lh "$RUN_DIR/"
echo ""
echo "=== Summary ==="
cat "$RUN_DIR/logs/summary_all_"*.json 2>/dev/null | python -m json.tool | head -60
JOBEOF

sbatch jobs/run_train_full.sh
```

**时间估算（单张 A100 80G）**：

| 阶段 | 每实验耗时 | 3 实验合计 |
|------|-----------|-----------|
| Phase 1 (llm / regular ≈200K) | ~25 min | ~50 min |
| Phase 1 (combined ≈400K) | ~45 min | — |
| T/T* Filter（compute embeddings） | ~5 min | ~15 min |
| Phase 2 × 4 losses（各≈100K, 3 epoch） | ~60 min | ~3h |
| **合计** | — | **~5–7h** |

> 建议申请 `--time=16:00:00`，留足 buffer。

### 6c — 拆分训练（时限受限）

如果集群单次作业时限 < 8h，可以只运行 **combined** 实验（包含所有数据，结果最全面）。

修改 `train_hard_neg.py` 中的 `DATASETS` 字典来单独运行某个实验（临时改法，不需要永久修改）：

```bash
# 用 sed 临时只跑 combined 实验（在作业脚本中）
python - <<'EOF'
import re
code = open("train_hard_neg.py").read()
# 只保留 combined 实验
new_datasets = '''DATASETS: dict[str, list[str]] = {
    "combined": [
        "data/stage2/processed/direct_negation_attack/LLM/constructed_data.json",
        "data/stage2/processed/direct_negation_attack/Regular/constructed_data.json",
    ],
}
'''
code = re.sub(r'DATASETS:.*?^}', new_datasets, code, flags=re.DOTALL | re.MULTILINE)
open("train_hard_neg_combined_only.py", "w").write(code)
print("Written train_hard_neg_combined_only.py")
EOF

python train_hard_neg_combined_only.py
```

或者提交三个独立作业（各只跑一个实验），每个作业约 2h。

---

## 8. Step 7：评估作业

### 7a — 复现 Baseline（首次必跑）

先生成 `baseline_reference.json`，确认评估管道与 MTEB 排行榜对齐：

```bash
cat > jobs/run_verify_baseline.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=verify_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_baseline_%j.out
#SBATCH --error=logs/verify_baseline_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg
HF_CACHE=/path/to/shared/.cache/huggingface

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hard_neg
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1

echo "Start: $(date)"
python evaluate/verify_baseline.py --generate
echo "Done: $(date)"
echo "Reference saved to evaluate/baseline_reference.json"
cat evaluate/baseline_reference.json
JOBEOF

sbatch jobs/run_verify_baseline.sh
```

成功后 `evaluate/baseline_reference.json` 会存储 all-MiniLM-L6-v2 在 10 个 STS 任务上的参考分数。

### 7b — 对比评估训练后模型

训练结束后，对你关心的模型（例如 `combined/phase2_triplet_cascade/final`）运行对比评估：

```bash
cat > jobs/run_eval.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg
HF_CACHE=/path/to/shared/.cache/huggingface

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hard_neg
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1

echo "Start: $(date)"

# ── 修改这里：指定你要评估的模型路径 ─────────────────────
# 从 models/ 目录找到最新的训练输出
RUN_DIR=$(ls -td models/*_compare | grep -v debug | grep -v test | head -1)
echo "Evaluating run: $RUN_DIR"

# 评估所有 Phase 2 最终模型，与 baseline 对比
for exp in llm regular combined; do
    for loss in triplet_cascade batch_hard batch_semi_hard batch_hard_soft_margin; do
        MODEL_PATH="$RUN_DIR/$exp/phase2_$loss/final"
        if [ -d "$MODEL_PATH" ]; then
            echo ""
            echo "=== $exp / $loss ==="
            python evaluate/verify_baseline.py --check --model "$MODEL_PATH"
        fi
    done
done

echo ""
echo "Done: $(date)"
JOBEOF

sbatch jobs/run_eval.sh
```

**每个模型评估约 30–45 分钟**（10 个 MTEB STS 任务），12 个模型共需 ~6–9h。
建议先只评估你最关心的模型，用 `--model` 直接指定路径。

### 7c — 手动评估单个模型（推荐先用这个）

```bash
# 在登录节点交互或提交短作业
MODEL="models/<RUN_ID>_compare/combined/phase2_triplet_cascade/final"
python evaluate/verify_baseline.py --check --model "$MODEL"
```

---

## 9. Step 8：监控与进度追踪

### 查看队列

```bash
squeue -u $USER --format="%.10i %.25j %.8T %.10M %R"
```

### 实时跟踪训练日志

```bash
# 查看作业输出（最新 50 行，持续刷新）
tail -f logs/train_<job_id>.out

# 不占用终端的后台监控
watch -n 30 "tail -20 logs/train_<job_id>.out"
```

### 查看当前训练到哪个阶段

```bash
# 训练日志存在 models/ 内
tail -50 models/*_compare/logs/training_*.log 2>/dev/null | grep "\[P[12]\|Filter\|EXPERIMENT\|Phase"
```

期望看到类似：
```
[EXPERIMENT] LLM
[P1/llm] train=..., test=...
[Filter/llm] Computing T and T* ...
[Filter/llm] kept=... (xx%)
--- Phase 2 / llm / triplet_cascade ---
--- Phase 2 / llm / batch_hard ---
...
[EXPERIMENT] REGULAR
...
```

### 查看汇总结果（训练完成后）

```bash
RUN_DIR=$(ls -td models/*_compare | grep -v debug | grep -v test | head -1)
python - <<EOF
import json
data = json.load(open("$RUN_DIR/logs/summary_all_".__add__(
    __import__('os').listdir("$RUN_DIR/logs/")[0].split("summary_all_")[-1])
    if any("summary_all" in f for f in __import__('os').listdir("$RUN_DIR/logs/")) else "{}"))
for s in data:
    print(f"{s['exp_name']:<12} {s['loss_name']:<32} train={s['train_size']:>6} "
          f"P1-Hard={s['p1_hard']:.4f} P2-Hard={s['p2_hard']:.4f} Δ={s['delta_p2_hard']:+.4f}")
EOF
```

或者直接读 JSON：

```bash
cat models/*_compare/logs/summary_all_*.json | python -m json.tool
```

### GPU 使用情况

```bash
# ssh 到计算节点后
nvidia-smi -l 2    # 每 2 秒刷新
```

### 取消作业

```bash
scancel <job_id>
```

---

## 10. Step 9：下载结果到本地

```bash
# 在本地终端执行

# 下载训练汇总
scp your_username@your_hpc_host:/path/to/hard_neg/models/<RUN_ID>_compare/logs/summary_all_*.json ./

# 下载 MTEB 评估结果
scp -r your_username@your_hpc_host:/path/to/hard_neg/evaluate/results/ ./evaluate/

# 下载 baseline 参考文件
scp your_username@your_hpc_host:/path/to/hard_neg/evaluate/baseline_reference.json ./evaluate/

# 如果需要下载某个模型权重（体积约 90MB）
rsync -avz --progress \
    your_username@your_hpc_host:/path/to/hard_neg/models/<RUN_ID>_compare/combined/phase2_triplet_cascade/final/ \
    ./models/best_model/

# 下载过滤统计
scp your_username@your_hpc_host:/path/to/hard_neg/models/<RUN_ID>_compare/filtered_data/filter_stats_*.json ./
```

---

## 11. 常见报错排查

### ImportError: cannot import name 'BatchHardTripletLoss'

```
ImportError: cannot import name 'BatchHardTripletLoss' from 'sentence_transformers.losses'
```

sentence-transformers 版本太低。

```bash
pip install "sentence-transformers>=5.1.2"
```

### CUDA out of memory（训练阶段）

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

修改 `train_hard_neg.py` 中两处 batch_size：

```python
# Phase 1
per_device_train_batch_size = 8   # 原 16，减半

# Phase 2
per_device_train_batch_size = 16  # 原 32，减半
```

### 计算节点无法访问 HuggingFace

```
OSError: We couldn't connect to 'https://huggingface.co'
```

在作业脚本中加入：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

同时确认已在登录节点提前下载了所有模型（Step 4）。

### conda activate 在 sbatch 中失效

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
```

作业脚本必须使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hard_neg
```

不能直接写 `conda activate hard_neg`（sbatch 非交互式 shell 中无效）。

### 数据文件不存在

```
AssertionError: 数据文件不存在: data/stage2/processed/direct_negation_attack/LLM/constructed_data.json
```

训练代码加载三个实验的数据：

```
data/stage2/processed/direct_negation_attack/LLM/constructed_data.json
data/stage2/processed/direct_negation_attack/Regular/constructed_data.json
```

确认这两个文件存在后再提交训练作业：

```bash
ls -lh data/stage2/processed/direct_negation_attack/LLM/constructed_data.json
ls -lh data/stage2/processed/direct_negation_attack/Regular/constructed_data.json
```

### spaCy 模型找不到（构建阶段）

```
OSError: [E050] Can't find model 'en_core_web_sm'
```

```bash
conda activate hard_neg
python -m spacy download en_core_web_sm
# 无网络时：
# pip download en-core-web-sm --no-deps -d /tmp/spacy_pkg/
# pip install /tmp/spacy_pkg/en_core_web_sm-*.whl
```

### 作业超时

```bash
# 查看已跑到哪个实验 / phase
grep "EXPERIMENT\|Phase 2\|Filter\|DONE" logs/train_<job_id>.out | tail -20

# 查看已有哪些 checkpoint 保存
find models/*_compare -name "config.json" -path "*/final/*" | sort
```

超时后可以手动运行还未完成的实验（通过临时修改 `DATASETS` 只保留未完成的实验，见 Step 6c）。

### 查看作业失败原因

```bash
cat logs/train_<job_id>.err
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

---

## 快速参考：作业提交顺序

```bash
# 1. 验证流程正确（先跑）
sbatch jobs/run_train_test.sh        # ~30min

# 2. 查看 test 结果
tail -30 logs/train_test_<job_id>.out

# 3. 确认无误后提交正式训练
sbatch jobs/run_train_full.sh        # ~6-8h

# 4. 训练完成后，先生成 baseline
sbatch jobs/run_verify_baseline.sh   # ~45min（仅首次需要）

# 5. 评估训练好的模型
sbatch jobs/run_eval.sh              # 按需调整模型路径
```
