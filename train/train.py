import os
import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers import TrainerCallback

# ─────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────
# 调试模式：python train_hard_neg.py --debug 只用每组前100条数据快速验证
DEBUG = "--debug" in sys.argv
DEBUG_SIZE = 100

RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_debug" if DEBUG else "")
MODEL_NAME = "all-MiniLM-L6-v2"

# 三组实验的数据路径（success=True 的行）
DATASETS = {
    "llm": [
        "data/stage2/processed/direct_negation_attack/LLM/constructed_data.json",
    ],
    "regular": [
        "data/stage2/processed/direct_negation_attack/Regular/constructed_data.json",
    ],
    "combined": [
        "data/stage2/processed/direct_negation_attack/LLM/constructed_data.json",
        "data/stage2/processed/direct_negation_attack/Regular/constructed_data.json",
    ],
}

# 全局日志目录（三个实验共用）
GLOBAL_LOG_DIR = Path(f"models/{RUN_ID}_compare/logs")
GLOBAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = GLOBAL_LOG_DIR / f"training_{RUN_ID}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Run ID     : {RUN_ID}")
logger.info(f"Output base: models/{RUN_ID}_compare/")


# ─────────────────────────────────────────────
# 1. Callback：记录 train_loss + eval 指标
# ─────────────────────────────────────────────
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, csv_path: Path, json_path: Path):
        self.csv_path  = csv_path
        self.json_path = json_path
        self.all_metrics: list[dict] = []

    def _record(self, row: dict):
        self.all_metrics.append(row)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.all_metrics, f, ensure_ascii=False, indent=2)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4)}
        row.update({k: round(v, 6) if isinstance(v, float) else v
                    for k, v in logs.items()})
        self._record(row)
        logger.info(f"[Step {state.global_step}] {logs}")

    def on_train_end(self, args, state, control, **kwargs):
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in self.all_metrics:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(self.all_metrics)
        logger.info(f"Metrics saved → {self.csv_path}")


# ─────────────────────────────────────────────
# 2. 加载本地 constructed_data.json
#
#    字段：id, anchor, pos, neg, hard_neg,
#          method, recognizer, success, ...
#
#    过滤 success=True，映射：
#      pos       → positive
#      hard_neg  → negative
#      neg       → neg_orig（评估用）
# ─────────────────────────────────────────────
def load_from_json(paths: list[str]) -> Dataset:
    """从多个 constructed_data.json 加载，过滤 success=True，去重（同 id 保留第一条）。"""
    seen_ids: set[str] = set()
    rows: list[dict] = []
    for path in paths:
        assert Path(path).exists(), f"数据文件不存在: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if not item.get("success") or not item.get("hard_neg"):
                continue
            rid = item["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rows.append({
                "id":        rid,
                "anchor":    item["anchor"],
                "positive":  item["pos"],
                "negative":  item["hard_neg"],   # 训练用
                "neg_orig":  item["neg"],         # 评估 A 用
                "recognizer": item["recognizer"],
            })
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────
# 3. 构建共享测试集（从 combined 数据切分，三个实验共用）
#    保证比较的公平性：所有模型在完全相同的测试样本上评估
# ─────────────────────────────────────────────
logger.info("Building shared test set from combined data ...")
combined_full = load_from_json(DATASETS["combined"])
logger.info(f"Combined total (success): {len(combined_full)}")

split_combined = combined_full.train_test_split(test_size=0.1, seed=42)
shared_test    = split_combined["test"]
shared_test_ids: set[str] = set(shared_test["id"])
logger.info(f"Shared test set: {len(shared_test)} samples")

# 评估集 A：原始负样本（anchor + pos + neg）
eval_orig = shared_test.select_columns(["anchor", "positive", "neg_orig"]) \
                        .rename_columns({"neg_orig": "negative"})

# 评估集 B：困难负样本（anchor + pos + hard_neg）
eval_hard = shared_test.select_columns(["anchor", "positive", "negative"])


# ─────────────────────────────────────────────
# 4. 绘图函数（供每个实验调用）
# ─────────────────────────────────────────────
def extract_series(metrics: list[dict], key: str):
    steps, values = [], []
    for row in metrics:
        if key in row and row[key] is not None:
            steps.append(row["step"])
            values.append(row[key])
    return steps, values


def get_accuracy(results: dict) -> float:
    for k, v in results.items():
        if "accuracy" in k.lower():
            return float(v)
    return 0.0


def plot_all(metrics, base_orig, base_hard, final_orig, final_hard, save_dir, run_label):
    C_BLUE  = "#378ADD"
    C_TEAL  = "#1D9E75"
    C_AMBER = "#BA7517"
    C_GRAY  = "#888780"
    C_RED   = "#D94F3D"
    C_BG    = "#FFFFFF"
    C_GRID  = "#EBEBEB"

    plt.rcParams.update({
        "figure.facecolor": C_BG, "axes.facecolor": C_BG,
        "axes.edgecolor": C_GRAY, "axes.grid": True,
        "grid.color": C_GRID, "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(14, 10), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A：Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    steps_loss, vals_loss = extract_series(metrics, "loss")
    if steps_loss:
        ax1.plot(steps_loss, vals_loss, color=C_BLUE, linewidth=1.8,
                 marker="o", markersize=3, label="train loss")
        if len(vals_loss) >= 4:
            z  = np.polyfit(steps_loss, vals_loss, 2)
            xs = np.linspace(min(steps_loss), max(steps_loss), 200)
            ax1.plot(xs, np.poly1d(z)(xs), color=C_AMBER, linewidth=1.2,
                     linestyle="--", label="trend")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=8)

    # Panel B：Eval Loss（困难负样本集上）
    ax2 = fig.add_subplot(gs[0, 1])
    steps_eval, vals_eval = extract_series(metrics, "eval_loss")
    if steps_eval:
        ax2.plot(steps_eval, vals_eval, color=C_TEAL, linewidth=1.8,
                 marker="s", markersize=4, label="eval loss (hard neg)")
        ax2.set_title("Eval Loss (Hard Negatives)")
        ax2.set_xlabel("Step"); ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No eval loss recorded",
                 ha="center", va="center", color=C_GRAY,
                 transform=ax2.transAxes)
        ax2.set_title("Eval Loss (Hard Negatives)")

    # Panel C：Triplet Accuracy — 原始负样本 vs 困难负样本（Base + Fine-tuned）
    ax3 = fig.add_subplot(gs[1, 0])
    categories  = ["Orig Neg\n(Base)", "Orig Neg\n(FT)", "Hard Neg\n(Base)", "Hard Neg\n(FT)"]
    acc_values  = [get_accuracy(base_orig), get_accuracy(final_orig),
                   get_accuracy(base_hard),  get_accuracy(final_hard)]
    bar_colors  = [C_GRAY, C_BLUE, C_GRAY, C_RED]
    bars = ax3.bar(categories, acc_values, color=bar_colors,
                   width=0.5, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, acc_values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Triplet Accuracy Comparison")
    ax3.set_ylabel("Accuracy")
    delta_orig = get_accuracy(final_orig) - get_accuracy(base_orig)
    delta_hard = get_accuracy(final_hard) - get_accuracy(base_hard)
    ax3.annotate(f"Δ={delta_orig:+.4f}", xy=(0.5, max(acc_values[:2]) + 0.04),
                 ha="center", fontsize=8, color=C_BLUE)
    ax3.annotate(f"Δ={delta_hard:+.4f}", xy=(2.5, max(acc_values[2:]) + 0.04),
                 ha="center", fontsize=8, color=C_RED)

    # Panel D：所有 eval 指标对比（orig neg 和 hard neg 各自的提升）
    ax4 = fig.add_subplot(gs[1, 1])
    common_keys = sorted(
        k for k in base_orig
        if k in final_orig and isinstance(base_orig[k], (int, float))
    )
    if common_keys:
        x = np.arange(len(common_keys))
        w = 0.2
        base_o  = [float(base_orig[k])  for k in common_keys]
        final_o = [float(final_orig[k]) for k in common_keys]
        base_h  = [float(base_hard.get(k, 0)) for k in common_keys]
        final_h = [float(final_hard.get(k, 0)) for k in common_keys]
        ax4.bar(x - 1.5*w, base_o,  w, label="Base/Orig",  color=C_GRAY,  alpha=0.8)
        ax4.bar(x - 0.5*w, final_o, w, label="FT/Orig",    color=C_BLUE,  alpha=0.8)
        ax4.bar(x + 0.5*w, base_h,  w, label="Base/Hard",  color=C_GRAY,  alpha=0.5)
        ax4.bar(x + 1.5*w, final_h, w, label="FT/Hard",    color=C_RED,   alpha=0.8)
        short = [k.split("_", 2)[-1] for k in common_keys]
        ax4.set_xticks(x)
        ax4.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
        ax4.set_title("All Metrics: Orig vs Hard Neg")
        ax4.set_ylabel("Score")
        ax4.legend(fontsize=7, ncol=2)
    else:
        ax4.text(0.5, 0.5, "No comparable metrics",
                 ha="center", va="center", color=C_GRAY,
                 transform=ax4.transAxes)
        ax4.set_title("All Metrics: Orig vs Hard Neg")

    fig.suptitle(f"Training Summary  |  {run_label}",
                 fontsize=13, fontweight="bold", y=1.01)

    path = save_dir / f"training_summary_{run_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    logger.info(f"Plot saved → {path}")
    return path


# ─────────────────────────────────────────────
# 5. 单次实验函数
#    name         : "llm" / "regular" / "combined"
#    data_paths   : 该实验的 constructed_data.json 路径列表
#    eval_orig_ds : 共享测试集（原始负样本）
#    eval_hard_ds : 共享测试集（困难负样本）
# ─────────────────────────────────────────────
def run_experiment(
    name: str,
    data_paths: list[str],
    eval_orig_ds: Dataset,
    eval_hard_ds: Dataset,
) -> dict:
    run_label = f"{RUN_ID}_{name}"
    logger.info("=" * 60)
    logger.info(f"[EXPERIMENT] {name.upper()}")
    logger.info("=" * 60)

    # ── 输出目录 ──────────────────────────────────────────────────────────
    output_dir = Path(f"models/{RUN_ID}_compare/{name}")
    final_dir  = output_dir / "final"
    log_dir    = output_dir / "logs"
    plot_dir   = output_dir / "plots"
    for d in [output_dir, final_dir, log_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_file  = log_dir / f"metrics_{run_label}.csv"
    json_file = log_dir / f"metrics_{run_label}.json"

    # ── 加载训练数据（排除共享测试集中的 ID）──────────────────────────────
    full_ds = load_from_json(data_paths)
    train_ds = full_ds.filter(lambda x: x["id"] not in shared_test_ids)
    if DEBUG:
        train_ds = train_ds.select(range(min(DEBUG_SIZE, len(train_ds))))
        logger.info(f"  [DEBUG] train truncated to {len(train_ds)} samples")
    train_dataset = train_ds.select_columns(["anchor", "positive", "negative"])

    recognizer_dist = Counter(full_ds["recognizer"])
    logger.info(f"  Data source  : {data_paths}")
    logger.info(f"  Total success: {len(full_ds)}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Test  samples: {len(eval_hard_ds)} (shared)")
    logger.info(f"  Recognizer   : {dict(recognizer_dist)}")

    # ── 加载模型（每次实验独立加载，保证公平比较）─────────────────────────
    logger.info(f"  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── 评估基础模型（训练前）──────────────────────────────────────────────
    evaluator_orig = TripletEvaluator(
        anchors=eval_orig_ds["anchor"],
        positives=eval_orig_ds["positive"],
        negatives=eval_orig_ds["negative"],
        name=f"{name}-eval-orig-neg",
    )
    evaluator_hard = TripletEvaluator(
        anchors=eval_hard_ds["anchor"],
        positives=eval_hard_ds["positive"],
        negatives=eval_hard_ds["negative"],
        name=f"{name}-eval-hard-neg",
    )

    logger.info("  Evaluating base model (before training)...")
    base_orig = evaluator_orig(model)
    base_hard = evaluator_hard(model)
    logger.info(f"  Base | orig neg: {base_orig}")
    logger.info(f"  Base | hard neg: {base_hard}")

    # ── 损失函数 & 训练参数 ────────────────────────────────────────────────
    loss             = MultipleNegativesRankingLoss(model)
    metrics_callback = MetricsLoggerCallback(csv_file, json_file)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        run_name=f"mnrl-hard-neg_{run_label}",
    )

    # ── 训练 ───────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_hard_ds,      # 用困难负样本做 in-training eval
        loss=loss,
        evaluator=evaluator_hard,       # checkpoint 保存时触发
        callbacks=[metrics_callback],
    )

    logger.info("  Starting training...")
    train_result = trainer.train()
    logger.info(f"  Training finished. {train_result.metrics}")

    # ── 评估训练后模型 ────────────────────────────────────────────────────
    logger.info("  Evaluating fine-tuned model...")
    final_orig = evaluator_orig(model)
    final_hard = evaluator_hard(model)
    logger.info(f"  Fine-tuned | orig neg: {final_orig}")
    logger.info(f"  Fine-tuned | hard neg: {final_hard}")

    # ── 保存对比结果 ──────────────────────────────────────────────────────
    comparison = {
        "run_id":        run_label,
        "experiment":    name,
        "data_paths":    data_paths,
        "train_samples": len(train_dataset),
        "test_samples":  len(eval_hard_ds),
        "recognizer_dist": dict(recognizer_dist),
        "base_model": {
            "orig_neg": base_orig,
            "hard_neg": base_hard,
        },
        "fine_tuned_model": {
            "orig_neg": final_orig,
            "hard_neg": final_hard,
        },
    }
    comparison_path = log_dir / f"comparison_{run_label}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"  Comparison saved → {comparison_path}")

    # ── 保存模型 ──────────────────────────────────────────────────────────
    logger.info(f"  Saving model → {final_dir}")
    model.save_pretrained(str(final_dir))

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plot_path = plot_all(
        metrics_callback.all_metrics,
        base_orig, base_hard,
        final_orig, final_hard,
        plot_dir,
        run_label,
    )

    # ── 摘要 ──────────────────────────────────────────────────────────────
    def acc(d): return get_accuracy(d)
    logger.info(f"  Train samples : {len(train_dataset):,}")
    logger.info(f"  Test  samples : {len(eval_hard_ds):,}")
    logger.info(f"  Orig  neg  acc: {acc(base_orig):.4f} → {acc(final_orig):.4f}  "
                f"(Δ {acc(final_orig)-acc(base_orig):+.4f})")
    logger.info(f"  Hard  neg  acc: {acc(base_hard):.4f} → {acc(final_hard):.4f}  "
                f"(Δ {acc(final_hard)-acc(base_hard):+.4f})")
    logger.info(f"  Model : {final_dir}")
    logger.info(f"  Plot  : {plot_path}")

    return {
        "name":        name,
        "train_size":  len(train_dataset),
        "base_orig":   acc(base_orig),
        "base_hard":   acc(base_hard),
        "final_orig":  acc(final_orig),
        "final_hard":  acc(final_hard),
        "delta_orig":  acc(final_orig) - acc(base_orig),
        "delta_hard":  acc(final_hard) - acc(base_hard),
    }


# ─────────────────────────────────────────────
# 6. 主循环：依次训练三个模型
# ─────────────────────────────────────────────
all_summaries: list[dict] = []

for exp_name, exp_paths in DATASETS.items():
    summary = run_experiment(
        name=exp_name,
        data_paths=exp_paths,
        eval_orig_ds=eval_orig,
        eval_hard_ds=eval_hard,
    )
    all_summaries.append(summary)


# ─────────────────────────────────────────────
# 7. 横向对比摘要
# ─────────────────────────────────────────────
summary_path = Path(f"models/{RUN_ID}_compare/logs/summary_all_{RUN_ID}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, ensure_ascii=False, indent=2)

logger.info("\n" + "=" * 60)
logger.info("ALL EXPERIMENTS COMPLETE")
logger.info("=" * 60)
logger.info(f"{'Experiment':<12} {'Train':>8} {'BaseOrig':>9} {'FTOrig':>8} "
            f"{'ΔOrig':>7} {'BaseHard':>9} {'FTHard':>8} {'ΔHard':>7}")
logger.info("-" * 70)
for s in all_summaries:
    logger.info(
        f"{s['name']:<12} {s['train_size']:>8,} "
        f"{s['base_orig']:>9.4f} {s['final_orig']:>8.4f} {s['delta_orig']:>+7.4f} "
        f"{s['base_hard']:>9.4f} {s['final_hard']:>8.4f} {s['delta_hard']:>+7.4f}"
    )
logger.info("=" * 60)
logger.info(f"Summary → {summary_path}")
