#!/bin/bash
#SBATCH -p debug # 指定GPU队列
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -J angle——loss_abl
#SBATCH -n 8            # 指定CPU总核心数
#SBATCH --gres=gpu:1   # 指定GPU卡数
#SBATCH -D results/angle/330

source ~/.bashrc
conda activate rpl
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

# 确保当前目录正确（在提交目录）
cd $SLURM_SUBMIT_DIR

echo "Job started at $(date)"


python scripts/run_pipeline.py \
  --input data/test.jsonl \
  --out_dir outputs/run_2 \
  --evaluate



echo "Job ended at $(date)"