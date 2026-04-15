# HPC 全量数据集构造指引

从零开始，在远程 HPC 集群上完成 Hard Negative 全量数据集构造的端到端操作手册。

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [Step 1：登录节点 — 克隆代码](#2-step-1登录节点--克隆代码)
3. [Step 2：环境配置](#3-step-2环境配置)
4. [Step 3：数据准备](#4-step-3数据准备)
5. [Step 4：预下载 Qwen3 模型](#5-step-4预下载-qwen3-模型)
6. [Step 5：提交作业](#6-step-5提交作业)
   - [方案 A — Regular + LLM 一次跑完](#方案-a--regular--llm-一次跑完推荐)
   - [方案 B — Regular 和 LLM 分开提交](#方案-b--regular-和-llm-分开提交时限受限时使用)
7. [Step 6：监控作业](#7-step-6监控作业)
8. [Step 7：验证结果](#8-step-7验证结果)
9. [Step 8：下载结果到本地](#9-step-8下载结果到本地)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 整体流程概览

```
登录节点
  │
  ├── git clone / git pull          ← 拉取最新代码
  ├── bash setup_env.sh hard_neg    ← 创建 conda 环境 + 安装依赖
  ├── 上传原始数据 parquet           ← 数据文件太大不入 git
  ├── 预下载 Qwen3-1.7B 权重        ← 计算节点无公网，必须提前下
  │
  └── sbatch jobs/run_*.sh          ← 提交到 SLURM 队列
        │
        └── 计算节点
              ├── Regular 路径（纯规则，CPU 即可）  ~8h / 275K 条
              └── LLM 路径（Qwen3 批推理，需 GPU）  ~5h（A100）
```

**预期产出**（274,951 条原始数据）：

| 路径 | 成功样本 | 时间（A100 80G） | 时间（V100 16G） |
|---|---|---|---|
| Regular | ~247,000 条（89.8%） | ~8h | ~8h（无 GPU 需求） |
| LLM | ~170,000 条（61.7%） | ~5h | ~19h |

---

## 2. Step 1：登录节点 — 克隆代码

```bash
# SSH 登录集群登录节点
ssh your_username@your_hpc_host

# 进入你的工作目录（替换为实际路径）
cd /path/to/your/workspace

# 首次克隆
git clone https://github.com/Fu-fangteng/hard_negative_constuction.git hard_neg
cd hard_neg

# 如果已有旧版本，拉取最新代码
# cd hard_neg && git pull origin main
```

确认代码是最新的：

```bash
git log --oneline -5
# 应能看到最新提交，包含 "feat: batch LLM inference" 等
```

---

## 3. Step 2：环境配置

项目提供了一键配置脚本 `setup_env.sh`，自动完成以下操作：
- 创建 `hard_neg` conda 环境（Python 3.10）
- 安装 PyTorch（自动匹配 CUDA 版本）
- 安装全部依赖（transformers、spaCy 等）
- 下载 spaCy 英文模型 `en_core_web_sm`

```bash
# 在登录节点执行（需要 conda 已激活）
bash setup_env.sh hard_neg

# 激活环境
conda activate hard_neg
```

**手动验证环境是否正常**：

```bash
python - <<'EOF'
import torch, transformers, spacy, pandas, pyarrow
print(f"torch        : {torch.__version__}")
print(f"CUDA 可用    : {torch.cuda.is_available()}")
print(f"transformers : {transformers.__version__}")
print(f"spacy        : {spacy.__version__}")
print(f"pandas       : {pandas.__version__}")
nlp = spacy.load("en_core_web_sm")
print("en_core_web_sm : OK")
EOF
```

期望输出示例：
```
torch        : 2.x.x
CUDA 可用    : True
transformers : 4.x.x
spacy        : 3.x.x
pandas       : 2.x.x
en_core_web_sm : OK
```

> 如果 `CUDA 可用 : False`，检查 `module load cuda` 或联系管理员确认 GPU 驱动。

---

## 4. Step 3：数据准备

原始数据文件不入 git（体积大），需手动上传。

### 方式一：从本地 scp 上传

```bash
# 在本地终端执行
scp /path/to/hard_neg/data/raw/train-00000-of-00001.parquet \
    your_username@your_hpc_host:/path/to/workspace/hard_neg/data/raw/
```

### 方式二：集群上直接下载（如有公网）

```bash
# 在登录节点执行
mkdir -p data/raw
conda activate hard_neg

python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("sentence-transformers/nli-for-simcse", split="train")
ds.to_parquet("data/raw/train-00000-of-00001.parquet")
print(f"Downloaded {len(ds)} rows")
EOF
```

验证数据文件存在：

```bash
ls -lh data/raw/train-00000-of-00001.parquet
# 期望：约 60-80MB

python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/raw/train-00000-of-00001.parquet")
print(f"总行数   : {len(df):,}")
print(f"列名     : {df.columns.tolist()}")
print(df.head(2).to_string())
EOF
# 期望输出：总行数 : 274,951，列名 ['anchor', 'positive', 'negative']
```

---

## 5. Step 4：预下载 Qwen3 模型

> **仅 LLM 路径需要此步骤。** 如果只跑 Regular，可跳过。

计算节点通常无法访问互联网，必须在登录节点提前下载并缓存模型权重。

```bash
conda activate hard_neg

# 建议将缓存放在共享存储，避免每人重复下载
export HF_HOME=/path/to/shared/storage/.cache/huggingface
# 例如：export HF_HOME=/scratch/shared/.cache/huggingface

python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "Qwen/Qwen3-1.7B"

print(f"下载 tokenizer: {model_id} ...")
AutoTokenizer.from_pretrained(model_id)

print(f"下载模型权重: {model_id} ...")
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

print("下载完成，已缓存至:", __import__('os').environ.get('HF_HOME', '~/.cache/huggingface'))
EOF
```

验证模型可以加载（在登录节点，不需要 GPU）：

```bash
python - <<'EOF'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
print(f"词表大小: {tok.vocab_size}")
print("模型缓存验证 OK")
EOF
```

---

## 6. Step 5：提交作业

先创建作业脚本目录和日志目录：

```bash
cd /path/to/workspace/hard_neg
mkdir -p jobs logs
```

---

### 方案 A — Regular + LLM 一次跑完（推荐）

适合：单次作业时限 ≥ 24h、有 GPU 节点。

**创建作业脚本**：

```bash
cat > jobs/run_full.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_full
#SBATCH --partition=gpu              # ← 修改为你的 GPU 分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

set -e

# ── 路径配置（根据实际修改）────────────────────────────────
WORKDIR=/path/to/workspace/hard_neg        # ← 修改
HF_CACHE=/path/to/shared/.cache/huggingface  # ← 修改
CONDA_ENV=hard_neg

# ── 环境初始化 ────────────────────────────────────────────
cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"

echo "=========================================="
echo "作业 ID   : $SLURM_JOB_ID"
echo "节点      : $(hostname)"
echo "开始时间  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

# ── 自动根据显存选 batch_size ─────────────────────────────
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "$GPU_MEM" -ge 70000 ]; then BATCH=16
elif [ "$GPU_MEM" -ge 30000 ]; then BATCH=12
elif [ "$GPU_MEM" -ge 20000 ]; then BATCH=8
else BATCH=6
fi
echo "显存: ${GPU_MEM}MiB  → llm_batch_size=${BATCH}"

# ── 运行流水线 ────────────────────────────────────────────
python stage2/run_stage2.py \
    --input_path     data/raw/train-00000-of-00001.parquet \
    --output_base    data/stage2 \
    --methods        direct_negation_attack \
    --recognizer     both \
    --llm_model      Qwen/Qwen3-1.7B \
    --llm_batch_size "$BATCH"

echo "=========================================="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
JOBEOF
```

**提交作业**：

```bash
sbatch jobs/run_full.sh

# 输出示例：Submitted batch job 12345678
```

---

### 方案 B — Regular 和 LLM 分开提交（时限受限时使用）

#### B-1：Regular 路径（无 GPU，~8h）

```bash
cat > jobs/run_regular.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_regular
#SBATCH --partition=cpu              # ← CPU 分区即可
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/regular_%j.out
#SBATCH --error=logs/regular_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg   # ← 修改
CONDA_ENV=hard_neg

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python stage2/run_stage2.py \
    --input_path  data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --methods     direct_negation_attack \
    --recognizer  regular

echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
JOBEOF

sbatch jobs/run_regular.sh
```

#### B-2：LLM 路径（需要 GPU，A100 ~5h / V100 ~19h）

```bash
cat > jobs/run_llm.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=hard_neg_llm
#SBATCH --partition=gpu              # ← 修改为你的 GPU 分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00              # V100 需要更长时间
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

set -e

WORKDIR=/path/to/workspace/hard_neg        # ← 修改
HF_CACHE=/path/to/shared/.cache/huggingface  # ← 修改
CONDA_ENV=hard_neg

cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"

echo "节点: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "$GPU_MEM" -ge 70000 ]; then BATCH=16
elif [ "$GPU_MEM" -ge 30000 ]; then BATCH=12
elif [ "$GPU_MEM" -ge 20000 ]; then BATCH=8
else BATCH=6
fi
echo "显存: ${GPU_MEM}MiB  → llm_batch_size=${BATCH}"

python stage2/run_stage2.py \
    --input_path     data/raw/train-00000-of-00001.parquet \
    --output_base    data/stage2 \
    --methods        direct_negation_attack \
    --recognizer     llm \
    --llm_model      Qwen/Qwen3-1.7B \
    --llm_batch_size "$BATCH"

echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
JOBEOF

sbatch jobs/run_llm.sh
```

#### B-3：两个作业都完成后，合并生成 difference.md 和 final_dataset

```bash
conda activate hard_neg
python - << 'EOF'
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
from stage2.analyzer import aggregate_final_dataset, generate_difference_report
from stage2.builder import RunResult

base = Path("data/stage2/processed/direct_negation_attack")

def load_result(recognizer):
    d = base / recognizer
    records = json.loads((d / "constructed_data.json").read_text())
    stats   = json.loads((d / "method_stat.json").read_text())
    feat    = {r["id"]: 0 for r in records}
    return RunResult(
        method_name="direct_negation_attack",
        recognizer_type=recognizer,
        records=records,
        stats=stats,
        feature_counts=feat,
    )

reg = load_result("Regular")
llm = load_result("LLM")
all_results = {"direct_negation_attack": {"Regular": reg, "LLM": llm}}

diff = generate_difference_report("direct_negation_attack", reg, llm)
(base / "difference.md").write_text(diff, encoding="utf-8")
print("✓ difference.md 写入完成")

final_rows = aggregate_final_dataset(all_results)
out = Path("data/stage2/processed/final_dataset.jsonl")
with out.open("w") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"✓ final_dataset.jsonl: {len(final_rows):,} 条")
EOF
```

---

## 7. Step 6：监控作业

### 查看队列状态

```bash
squeue -u $USER                        # 查看当前用户全部作业
squeue -j <job_id>                     # 查看指定作业
squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %R"  # 详细格式
```

状态说明：`PD`=排队中  `R`=运行中  `CG`=即将完成  `F`=失败

### 实时查看日志

```bash
# 持续刷新日志尾部
tail -f logs/full_<job_id>.out

# 每 30 秒刷新一次（不占终端）
watch -n 30 "tail -20 logs/full_<job_id>.out"
```

### 查看运行进度

```bash
# Regular 路径：已处理样本数（每条一行日志）
wc -l data/stage2/processed/direct_negation_attack/Regular/construction_log.jsonl

# LLM 路径：查看批处理进度
grep "batch\|Generating" logs/llm_<job_id>.out | tail -5

# 查看 GPU 使用情况（需 ssh 到计算节点）
ssh <compute_node_name>
nvidia-smi -l 2     # 每 2 秒刷新
```

### 取消作业

```bash
scancel <job_id>         # 取消单个作业
scancel -u $USER         # 取消该用户所有作业（谨慎！）
```

---

## 8. Step 7：验证结果

作业完成后，运行以下验证脚本：

```bash
conda activate hard_neg
python - << 'EOF'
import json, random
from pathlib import Path

base = Path("data/stage2/processed/direct_negation_attack")

print("=" * 50)
for recognizer in ["Regular", "LLM"]:
    stat_path = base / recognizer / "method_stat.json"
    if not stat_path.exists():
        print(f"[{recognizer}] ❌ method_stat.json 不存在")
        continue
    stat = json.loads(stat_path.read_text())
    ok = stat['success_ratio'] >= (0.85 if recognizer == "Regular" else 0.55)
    mark = "✓" if ok else "✗ 低于预期"
    print(f"\n[{recognizer}] {mark}")
    print(f"  总样本  : {stat['total_samples']:,}")
    print(f"  成功    : {stat['success_count']:,}  ({stat['success_ratio']*100:.1f}%)")
    print(f"  处理时间: {stat['processing_time_sec']}s")
    for reason, count in stat['failure_reasons'].items():
        if count:
            print(f"  失败-{reason}: {count:,}")

# final_dataset
print("\n" + "=" * 50)
final = Path("data/stage2/processed/final_dataset.jsonl")
if final.exists():
    rows = [json.loads(l) for l in final.read_text().splitlines() if l.strip()]
    ok = len(rows) >= 240000
    mark = "✓" if ok else "✗ 数量不足"
    print(f"\n[final_dataset] {mark}")
    print(f"  总行数: {len(rows):,}")
    print("\n  抽样展示（3 条）：")
    random.seed(42)
    for r in random.sample(rows, min(3, len(rows))):
        print(f"\n  anchor  : {r['anchor'][:80]}")
        print(f"  pos     : {r['pos']}")
        print(f"  hard_neg: {r['hard_neg']}")
        print(f"  method  : {r['method']} / {r['recognizer']}")
else:
    print("❌ final_dataset.jsonl 不存在，请运行合并步骤")
EOF
```

**成功标准**：

| 指标 | 期望值 |
|---|---|
| Regular 成功率 | ≥ 85% |
| LLM 成功率 | ≥ 55% |
| final_dataset 行数 | ≥ 240,000 |
| hard_neg 中不含 `Not X...` 开头的语法错误 | 0 条 |

---

## 9. Step 8：下载结果到本地

```bash
# 在本地终端执行
# 下载最终数据集（最重要）
scp your_username@your_hpc_host:/path/to/hard_neg/data/stage2/processed/final_dataset.jsonl \
    ./data/stage2/processed/

# 下载统计文件
scp -r your_username@your_hpc_host:/path/to/hard_neg/data/stage2/processed/direct_negation_attack/ \
    ./data/stage2/processed/

# 或用 rsync（支持断点续传，推荐）
rsync -avz --progress \
    your_username@your_hpc_host:/path/to/hard_neg/data/stage2/processed/ \
    ./data/stage2/processed/
```

---

## 10. 常见问题排查

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

```bash
# 调小 batch_size，每减半显存需求减半
--llm_batch_size 8   # 默认 16 → 改 8
--llm_batch_size 4   # 极端情况
```

### 计算节点无法访问 HuggingFace

```
OSError: We couldn't connect to 'https://huggingface.co'
```

在登录节点提前下载，然后作业脚本中指定本地路径：

```bash
# 查找本地缓存路径
find $HF_HOME -name "config.json" | grep Qwen3

# 在 sbatch 脚本中替换 --llm_model 参数
--llm_model /path/to/shared/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/<hash>/
```

### conda activate 在 sbatch 中失效

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
```

```bash
# 在 sbatch 脚本中用完整初始化方式
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hard_neg

# 而不是直接写
conda activate hard_neg   # ← sbatch 中此行无效
```

### 作业超时，如何确认已完成多少条

```bash
# 统计 Regular 路径已处理条数
DONE_REG=$(wc -l < data/stage2/processed/direct_negation_attack/Regular/construction_log.jsonl)

# 统计 LLM 路径已处理条数
DONE_LLM=$(wc -l < data/stage2/processed/direct_negation_attack/LLM/construction_log.jsonl)

echo "Regular: $DONE_REG / 274951"
echo "LLM    : $DONE_LLM / 274951"
```

> 当前版本不支持断点续传，若超时需重新完整运行。建议在申请时留足 buffer。

### spaCy 模型找不到

```
OSError: [E050] Can't find model 'en_core_web_sm'
```

```bash
conda activate hard_neg

# 有网络时重新下载
python -m spacy download en_core_web_sm

# 无网络时：在登录节点下载 whl 包后拷贝到计算节点安装
# pip download en-core-web-sm --no-deps -d /tmp/spacy_pkg/
# pip install /tmp/spacy_pkg/en_core_web_sm-*.whl
```

### 查看作业详细信息（失败原因）

```bash
# 查看完整的 stderr 输出
cat logs/full_<job_id>.err

# 查看作业资源使用情况
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```
