"""
train_hard_neg.py
=================
两阶段训练：Phase 1（MNRL 通用对齐）→ T/T* 筛选 → Phase 2（4 种 Loss 精调）

运行方式
────────
  python train_hard_neg.py              # 正式训练（全量数据）
  python train_hard_neg.py --test       # TEST 模式（每组截断到 TEST_SIZE，验证完整流程）
  python train_hard_neg.py --debug      # DEBUG 模式（极少量数据，仅验证代码不报错）

输出目录
────────
  models/{RUN_ID}_compare/
  ├── logs/                             # 全局日志
  ├── filtered_data/                    # 筛选结果（T/T* + case 字段）
  ├── {exp}/
  │   ├── phase1/final/                 # Phase 1 checkpoint
  │   ├── phase2_triplet_cascade/final/
  │   ├── phase2_batch_hard/final/
  │   ├── phase2_batch_semi_hard/final/
  │   └── phase2_batch_hard_soft_margin/final/
  └── summary_all_{RUN_ID}.json
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    TripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
)
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback

# ─────────────────────────────────────────────────────────────────────────────
# 0. 运行模式 & 全局配置
# ─────────────────────────────────────────────────────────────────────────────
DEBUG     = "--debug" in sys.argv
TEST_MODE = "--test"  in sys.argv

DEBUG_SIZE = 100
TEST_SIZE  = 500

_mode_tag  = "_debug" if DEBUG else ("_test" if TEST_MODE else "")
RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S") + _mode_tag
MODEL_NAME = "all-MiniLM-L6-v2"

DATASETS: dict[str, list[str]] = {
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

PHASE2_LOSSES = [
    "triplet_cascade",
    "batch_hard",
    "batch_semi_hard",
    "batch_hard_soft_margin",
]

FILTER_SAVE_DIR = Path(f"models/{RUN_ID}_compare/filtered_data")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 日志
# ─────────────────────────────────────────────────────────────────────────────
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
logger.info(f"Run ID   : {RUN_ID}  (debug={DEBUG}, test={TEST_MODE})")
logger.info(f"Output   : models/{RUN_ID}_compare/")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 工具函数
# ─────────────────────────────────────────────────────────────────────────────
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
        if not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4)}
        row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()})
        self._record(row)
        logger.info(f"[Step {state.global_step}] {logs}")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.all_metrics:
            return
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in self.all_metrics:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(self.all_metrics)
        logger.info(f"Metrics saved → {self.csv_path}")


def load_from_json(paths: list[str]) -> Dataset:
    """
    从 constructed_data.json 加载 success=True 的行。
    字段映射：pos→positive, hard_neg→negative, neg→neg_orig
    """
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
                "id":         rid,
                "anchor":     item["anchor"],
                "positive":   item["pos"],
                "negative":   item["hard_neg"],
                "neg_orig":   item["neg"],
                "method":     item.get("method", ""),
                "recognizer": item.get("recognizer", ""),
            })
    return Dataset.from_list(rows)


def get_accuracy(results: dict) -> float:
    for k, v in results.items():
        if "accuracy" in k.lower():
            return float(v)
    return 0.0


def _truncate_for_mode(ds: Dataset, label: str) -> Dataset:
    """根据运行模式截断数据集。"""
    if DEBUG:
        ds = ds.select(range(min(DEBUG_SIZE, len(ds))))
        logger.info(f"  [DEBUG] {label} truncated to {len(ds)}")
    elif TEST_MODE:
        ds = ds.select(range(min(TEST_SIZE, len(ds))))
        logger.info(f"  [TEST]  {label} truncated to {len(ds)}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 3. 共享测试集（所有实验共用，保证对比公平）
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Building shared test set from combined data ...")
combined_full = load_from_json(DATASETS["combined"])
logger.info(f"Combined total (success): {len(combined_full)}")

split_combined  = combined_full.train_test_split(test_size=0.1, seed=42)
shared_test     = split_combined["test"]
shared_test_ids: set[str] = set(shared_test["id"])
logger.info(f"Shared test set: {len(shared_test)} samples")

eval_orig = (shared_test
             .select_columns(["anchor", "positive", "neg_orig"])
             .rename_columns({"neg_orig": "negative"}))
eval_hard = shared_test.select_columns(["anchor", "positive", "negative"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase 1 绘图
# ─────────────────────────────────────────────────────────────────────────────
def plot_phase1(metrics, base_orig, base_hard, final_orig, final_hard,
                save_dir: Path, run_label: str) -> Path:
    C = {"blue": "#378ADD", "teal": "#1D9E75", "amber": "#BA7517",
         "gray": "#888780", "red": "#D94F3D", "bg": "#FFFFFF", "grid": "#EBEBEB"}
    plt.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
        "axes.edgecolor": C["gray"], "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig = plt.figure(figsize=(14, 10), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def series(key):
        s, v = [], []
        for r in metrics:
            if key in r and r[key] is not None:
                s.append(r["step"]); v.append(r[key])
        return s, v

    ax1 = fig.add_subplot(gs[0, 0])
    sl, vl = series("loss")
    if sl:
        ax1.plot(sl, vl, color=C["blue"], linewidth=1.8, marker="o", markersize=3, label="train loss")
        if len(vl) >= 4:
            xs = np.linspace(min(sl), max(sl), 200)
            ax1.plot(xs, np.poly1d(np.polyfit(sl, vl, 2))(xs),
                     color=C["amber"], linewidth=1.2, linestyle="--", label="trend")
        ax1.set_title("Training Loss"); ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    se, ve = series("eval_loss")
    if se:
        ax2.plot(se, ve, color=C["teal"], linewidth=1.8, marker="s", markersize=4,
                 label="eval loss (hard neg)")
        ax2.set_title("Eval Loss (Hard Negatives)"); ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No eval loss recorded", ha="center", va="center",
                 color=C["gray"], transform=ax2.transAxes)
        ax2.set_title("Eval Loss (Hard Negatives)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Loss")

    ax3 = fig.add_subplot(gs[1, 0])
    cats = ["Orig(Base)", "Orig(FT)", "Hard(Base)", "Hard(FT)"]
    vals = [get_accuracy(base_orig), get_accuracy(final_orig),
            get_accuracy(base_hard),  get_accuracy(final_hard)]
    bars = ax3.bar(cats, vals, color=[C["gray"], C["blue"], C["gray"], C["red"]],
                   width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylim(0, 1.1); ax3.set_title("Triplet Accuracy (Phase 1)")
    d_o = vals[1] - vals[0]; d_h = vals[3] - vals[2]
    ax3.annotate(f"Δ={d_o:+.4f}", xy=(0.5, max(vals[:2]) + 0.04), ha="center",
                 fontsize=8, color=C["blue"])
    ax3.annotate(f"Δ={d_h:+.4f}", xy=(2.5, max(vals[2:]) + 0.04), ha="center",
                 fontsize=8, color=C["red"])

    ax4 = fig.add_subplot(gs[1, 1])
    common = sorted(k for k in base_orig if k in final_orig
                    and isinstance(base_orig[k], (int, float)))
    if common:
        x = np.arange(len(common)); w = 0.2
        ax4.bar(x - 1.5*w, [float(base_orig[k]) for k in common],  w, label="Base/Orig",  color=C["gray"],  alpha=0.8)
        ax4.bar(x - 0.5*w, [float(final_orig[k]) for k in common], w, label="FT/Orig",    color=C["blue"],  alpha=0.8)
        ax4.bar(x + 0.5*w, [float(base_hard.get(k, 0)) for k in common],  w, label="Base/Hard", color=C["gray"],  alpha=0.5)
        ax4.bar(x + 1.5*w, [float(final_hard.get(k, 0)) for k in common], w, label="FT/Hard",   color=C["red"],   alpha=0.8)
        ax4.set_xticks(x); ax4.set_xticklabels([k.split("_", 2)[-1] for k in common],
                                                rotation=20, ha="right", fontsize=8)
        ax4.set_title("All Metrics Comparison"); ax4.legend(fontsize=7, ncol=2)
    else:
        ax4.text(0.5, 0.5, "No comparable metrics", ha="center", va="center",
                 color=C["gray"], transform=ax4.transAxes)
        ax4.set_title("All Metrics Comparison")

    fig.suptitle(f"Phase 1 Training Summary  |  {run_label}",
                 fontsize=13, fontweight="bold", y=1.01)
    path = save_dir / f"phase1_summary_{run_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info(f"Plot saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Phase 1：MNRL 通用语义对齐
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1(
    name: str,
    data_paths: list[str],
    eval_orig_ds: Dataset,
    eval_hard_ds: Dataset,
) -> tuple[SentenceTransformer, dict]:
    """
    Phase 1：MultipleNegativesRankingLoss，1 epoch。
    返回 (trained_model, results_dict)，results_dict 中有 p1_orig / p1_hard。
    """
    run_label  = f"{RUN_ID}_{name}_phase1"
    output_dir = Path(f"models/{RUN_ID}_compare/{name}/phase1")
    final_dir  = output_dir / "final"
    log_dir    = output_dir / "logs"
    plot_dir   = output_dir / "plots"
    for d in [output_dir, final_dir, log_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_file  = log_dir / f"metrics_{run_label}.csv"
    json_file = log_dir / f"metrics_{run_label}.json"

    full_ds  = load_from_json(data_paths)
    train_ds = full_ds.filter(lambda x: x["id"] not in shared_test_ids)
    train_ds = _truncate_for_mode(train_ds, f"{name}/phase1 train")
    train_dataset = train_ds.select_columns(["anchor", "positive", "negative"])

    logger.info(f"  [P1/{name}] train={len(train_dataset)}, test={len(eval_hard_ds)} (shared)")
    logger.info(f"  [P1/{name}] recognizer: {dict(Counter(full_ds['recognizer']))}")

    model = SentenceTransformer(MODEL_NAME)

    evaluator_orig = TripletEvaluator(
        anchors=eval_orig_ds["anchor"], positives=eval_orig_ds["positive"],
        negatives=eval_orig_ds["negative"], name=f"{name}-p1-orig",
    )
    evaluator_hard = TripletEvaluator(
        anchors=eval_hard_ds["anchor"], positives=eval_hard_ds["positive"],
        negatives=eval_hard_ds["negative"], name=f"{name}-p1-hard",
    )

    logger.info("  Evaluating base model before Phase 1 ...")
    base_orig = evaluator_orig(model)
    base_hard = evaluator_hard(model)
    logger.info(f"  Base | orig: {base_orig}")
    logger.info(f"  Base | hard: {base_hard}")

    loss             = MultipleNegativesRankingLoss(model)
    metrics_callback = MetricsLoggerCallback(csv_file, json_file)

    eval_steps = 100 if not (DEBUG or TEST_MODE) else 20
    args = SentenceTransformerTrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = 1,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        learning_rate               = 2e-5,
        warmup_ratio                = 0.1,
        fp16=True, bf16=False,
        batch_sampler               = BatchSamplers.NO_DUPLICATES,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 2,
        logging_steps               = max(eval_steps // 5, 10),
        run_name                    = run_label,
    )

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = args,
        train_dataset = train_dataset,
        eval_dataset  = eval_hard_ds,
        loss          = loss,
        evaluator     = evaluator_hard,
        callbacks     = [metrics_callback],
    )

    logger.info(f"  Starting Phase 1 training ({name}) ...")
    trainer.train()

    final_orig = evaluator_orig(model)
    final_hard = evaluator_hard(model)
    logger.info(f"  P1 done | orig: {get_accuracy(base_orig):.4f}→{get_accuracy(final_orig):.4f} "
                f"(Δ{get_accuracy(final_orig)-get_accuracy(base_orig):+.4f})")
    logger.info(f"  P1 done | hard: {get_accuracy(base_hard):.4f}→{get_accuracy(final_hard):.4f} "
                f"(Δ{get_accuracy(final_hard)-get_accuracy(base_hard):+.4f})")

    model.save_pretrained(str(final_dir))
    logger.info(f"  Phase 1 model → {final_dir}")

    plot_phase1(metrics_callback.all_metrics, base_orig, base_hard,
                final_orig, final_hard, plot_dir, run_label)

    p1_results = {
        "base_orig": base_orig,
        "base_hard": base_hard,
        "p1_orig":   final_orig,
        "p1_hard":   final_hard,
    }
    return model, p1_results


# ─────────────────────────────────────────────────────────────────────────────
# 6. 筛选：计算 T / T*，保留情况 B 和 C
# ─────────────────────────────────────────────────────────────────────────────
def compute_triplet_losses(
    model: SentenceTransformer,
    dataset: Dataset,
    margin: float = 0.2,
    batch_size: int = 64,
) -> Dataset:
    """
    对每条样本计算：
      T  = TripletLoss(anchor, pos, neg_orig)
      T* = TripletLoss(anchor, pos, hard_neg)
    返回新增 T / T_star / T_star_minus_T 字段的 Dataset。
    """
    model.eval()
    all_T, all_T_star = [], []

    anchors   = dataset["anchor"]
    positives = dataset["positive"]
    neg_origs = dataset["neg_orig"]
    hard_negs = dataset["negative"]

    with torch.no_grad():
        for i in range(0, len(anchors), batch_size):
            batch_a  = anchors[i   : i + batch_size]
            batch_p  = positives[i : i + batch_size]
            batch_n  = neg_origs[i : i + batch_size]
            batch_hn = hard_negs[i : i + batch_size]

            emb_a  = model.encode(batch_a,  convert_to_tensor=True, normalize_embeddings=True)
            emb_p  = model.encode(batch_p,  convert_to_tensor=True, normalize_embeddings=True)
            emb_n  = model.encode(batch_n,  convert_to_tensor=True, normalize_embeddings=True)
            emb_hn = model.encode(batch_hn, convert_to_tensor=True, normalize_embeddings=True)

            # cosine distance = 1 - cosine_similarity（归一化后等价）
            d_ap  = 1 - F.cosine_similarity(emb_a, emb_p)
            d_an  = 1 - F.cosine_similarity(emb_a, emb_n)
            d_ahn = 1 - F.cosine_similarity(emb_a, emb_hn)

            T      = torch.clamp(d_ap - d_an  + margin, min=0).cpu().tolist()
            T_star = torch.clamp(d_ap - d_ahn + margin, min=0).cpu().tolist()

            all_T.extend(T)
            all_T_star.extend(T_star)

    delta = [ts - t for ts, t in zip(all_T_star, all_T)]
    return (dataset
            .add_column("T",               all_T)
            .add_column("T_star",          all_T_star)
            .add_column("T_star_minus_T",  delta))


def filter_hard_negatives(
    dataset: Dataset,
    t_star_max: float = 2.0,
) -> tuple[Dataset, dict]:
    """
    筛选规则（只保留 B 和 C）：
      A：T ≈ 0 且 T* ≈ 0   → 丢弃（两个 neg 对模型都不难，无贡献）
      B：T ≈ 0 且 T* > 0   → 保留（hard_neg 比 neg_orig 更难，最优质）
      C：T > 0 且 T* > T   → 保留（hard_neg 有额外难度增量）
      D：T > 0 且 T* ≤ T   → 丢弃（hard_neg 不比 neg_orig 更难）
      E：T > 0 且 T* ≈ 0   → 丢弃（hard_neg 过于容易）
      overflow：T* > t_star_max → 丢弃（疑似构造错误）
    """
    total = len(dataset)
    stats = {"total": total, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
             "overflow": 0, "kept": 0}
    eps = 1e-6

    keep_rows = []
    for row in dataset:
        T, T_star, delta = row["T"], row["T_star"], row["T_star_minus_T"]

        if T_star > t_star_max:
            case = "overflow"
        elif T < eps and T_star < eps:
            case = "A"
        elif T < eps and T_star > eps:
            case = "B"
        elif T > eps and T_star > eps and delta > eps:
            case = "C"
        elif T > eps and T_star > eps and delta <= eps:
            case = "D"
        elif T > eps and T_star < eps:
            case = "E"
        else:
            case = "A"

        stats[case] = stats.get(case, 0) + 1
        if case in ("B", "C"):
            keep_rows.append({**row, "case": case})

    stats["kept"]       = len(keep_rows)
    stats["kept_ratio"] = round(stats["kept"] / total, 4) if total else 0.0

    return Dataset.from_list(keep_rows), stats


def save_filtered_dataset(
    dataset: Dataset,
    stats: dict,
    save_dir: Path,
    exp_name: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    data_path  = save_dir / f"filtered_data_{exp_name}.json"
    stats_path = save_dir / f"filter_stats_{exp_name}.json"

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(list(dataset), f, ensure_ascii=False, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"  Filtered data  → {data_path}")
    logger.info(f"  Filter stats   : {stats}")
    return data_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase 2：构建训练集 & 4 种 Loss 精调
# ─────────────────────────────────────────────────────────────────────────────
def build_train_datasets(
    filtered_ds: Dataset,
    loss_type: str,
) -> tuple:
    """
    根据 loss_type 返回 (train_dataset, batch_sampler)。
    - triplet_cascade    → {"base": ds_base, "hard": ds_hard}，NO_DUPLICATES
    - batch_hard / batch_semi_hard / batch_hard_soft_margin
                         → {"sentence": ..., "label": ...}，GROUP_BY_LABEL
    """
    if loss_type == "triplet_cascade":
        ds_base = Dataset.from_dict({
            "anchor":   filtered_ds["anchor"],
            "positive": filtered_ds["positive"],
            "negative": filtered_ds["neg_orig"],   # 原始 neg → base triplet
        })
        ds_hard = Dataset.from_dict({
            "anchor":   filtered_ds["anchor"],
            "positive": filtered_ds["positive"],
            "negative": filtered_ds["negative"],   # hard_neg → hard triplet
        })
        return {"base": ds_base, "hard": ds_hard}, BatchSamplers.NO_DUPLICATES

    else:
        # BatchHard* 系列：转为 (sentence, label) 格式
        # 每条样本：anchor + positive 同类（label=2i），hard_neg 异类（label=2i+1）
        sentences, labels = [], []
        for i, row in enumerate(filtered_ds):
            base_label = i * 2
            sentences += [row["anchor"], row["positive"], row["negative"]]
            labels    += [base_label,   base_label,       base_label + 1]

        label_ds = Dataset.from_dict({"sentence": sentences, "label": labels})
        return label_ds, BatchSamplers.GROUP_BY_LABEL


def run_phase2(
    name: str,
    loss_name: str,
    model_path: str,
    filtered_ds: Dataset,
    eval_orig_ds: Dataset,
    eval_hard_ds: Dataset,
    p1_results: dict,
) -> dict:
    """
    Phase 2：从 Phase 1 checkpoint 独立加载，用 loss_name 对应的 Loss 精调。
    每种 Loss 从同一 checkpoint 出发，保证对比公平。
    """
    run_label  = f"{RUN_ID}_{name}_phase2_{loss_name}"
    output_dir = Path(f"models/{RUN_ID}_compare/{name}/phase2_{loss_name}")
    final_dir  = output_dir / "final"
    log_dir    = output_dir / "logs"
    for d in [output_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_file  = log_dir / f"metrics_{run_label}.csv"
    json_file = log_dir / f"metrics_{run_label}.json"

    # 每种 Loss 从同一 Phase 1 checkpoint 独立加载
    model = SentenceTransformer(model_path)

    # 截断
    train_ds = _truncate_for_mode(filtered_ds, f"{name}/p2-{loss_name} train")

    logger.info(f"  [P2/{name}/{loss_name}] train={len(train_ds)} "
                f"(T_mean={sum(train_ds['T'])/max(len(train_ds),1):.4f}, "
                f"T*_mean={sum(train_ds['T_star'])/max(len(train_ds),1):.4f})")

    # 构建训练集 & Loss
    train_dataset, batch_sampler = build_train_datasets(train_ds, loss_name)

    if loss_name == "triplet_cascade":
        losses = {
            "base": TripletLoss(model, triplet_margin=0.2),
            "hard": TripletLoss(model, triplet_margin=0.2),
        }
        eval_dataset = {"hard": eval_hard_ds}
        metric_for_best = "eval_hard_loss"
        use_early_stop  = True
    elif loss_name == "batch_hard":
        losses = BatchHardTripletLoss(model, margin=0.5)
        eval_dataset    = None
        metric_for_best = None
        use_early_stop  = False
    elif loss_name == "batch_semi_hard":
        losses = BatchSemiHardTripletLoss(model, margin=0.5)
        eval_dataset    = None
        metric_for_best = None
        use_early_stop  = False
    elif loss_name == "batch_hard_soft_margin":
        losses = BatchHardSoftMarginTripletLoss(model)
        eval_dataset    = None
        metric_for_best = None
        use_early_stop  = False
    else:
        raise ValueError(f"Unknown loss_type: {loss_name}")

    evaluator_orig = TripletEvaluator(
        anchors=eval_orig_ds["anchor"], positives=eval_orig_ds["positive"],
        negatives=eval_orig_ds["negative"], name=f"{name}-p2-{loss_name}-orig",
    )
    evaluator_hard = TripletEvaluator(
        anchors=eval_hard_ds["anchor"], positives=eval_hard_ds["positive"],
        negatives=eval_hard_ds["negative"], name=f"{name}-p2-{loss_name}-hard",
    )

    metrics_callback = MetricsLoggerCallback(csv_file, json_file)

    eval_steps = 100 if not (DEBUG or TEST_MODE) else 20
    save_steps = eval_steps

    args = SentenceTransformerTrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = 3,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-6,
        warmup_ratio                = 0.1,
        fp16=True, bf16=False,
        batch_sampler               = batch_sampler,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = use_early_stop,
        metric_for_best_model       = metric_for_best,
        greater_is_better           = False if metric_for_best and "loss" in metric_for_best else None,
        logging_steps               = max(eval_steps // 5, 10),
        run_name                    = run_label,
    )

    callbacks = [metrics_callback]
    if use_early_stop:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        loss          = losses,
        evaluator     = evaluator_orig,   # 以 orig neg 不退化为保存标准
        callbacks     = callbacks,
    )

    logger.info(f"  Starting Phase 2 training ({name}/{loss_name}) ...")
    trainer.train()

    model.save_pretrained(str(final_dir))

    p2_orig = evaluator_orig(model)
    p2_hard = evaluator_hard(model)

    def _a(d): return get_accuracy(d)
    logger.info(f"  [P2/{loss_name}] orig: "
                f"{_a(p1_results['p1_orig']):.4f}→{_a(p2_orig):.4f} "
                f"(Δ{_a(p2_orig)-_a(p1_results['p1_orig']):+.4f})")
    logger.info(f"  [P2/{loss_name}] hard: "
                f"{_a(p1_results['p1_hard']):.4f}→{_a(p2_hard):.4f} "
                f"(Δ{_a(p2_hard)-_a(p1_results['p1_hard']):+.4f})")

    return {
        "exp_name":       name,
        "loss_name":      loss_name,
        "train_size":     len(train_ds),
        "p1_orig":        _a(p1_results["p1_orig"]),
        "p1_hard":        _a(p1_results["p1_hard"]),
        "p2_orig":        _a(p2_orig),
        "p2_hard":        _a(p2_hard),
        "delta_p2_orig":  _a(p2_orig) - _a(p1_results["p1_orig"]),
        "delta_p2_hard":  _a(p2_hard) - _a(p1_results["p1_hard"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. 主循环
# ─────────────────────────────────────────────────────────────────────────────
all_summaries: list[dict] = []

for exp_name, exp_paths in DATASETS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"[EXPERIMENT] {exp_name.upper()}")
    logger.info(f"{'='*60}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    p1_model, p1_results = run_phase1(
        name         = exp_name,
        data_paths   = exp_paths,
        eval_orig_ds = eval_orig,
        eval_hard_ds = eval_hard,
    )
    p1_model_path = str(Path(f"models/{RUN_ID}_compare/{exp_name}/phase1/final"))

    # ── 筛选：用 Phase 1 模型计算 T / T*，保留 B + C ─────────────────────────
    logger.info(f"\n  [Filter/{exp_name}] Computing T and T* ...")
    full_ds = load_from_json(exp_paths)
    full_ds = full_ds.filter(lambda x: x["id"] not in shared_test_ids)
    full_ds = _truncate_for_mode(full_ds, f"{exp_name}/filter input")

    full_ds_with_loss = compute_triplet_losses(p1_model, full_ds)

    filtered_ds, filter_stats = filter_hard_negatives(full_ds_with_loss)
    logger.info(f"  [Filter/{exp_name}] {filter_stats['total']} → "
                f"{filter_stats['kept']} kept ({filter_stats['kept_ratio']:.1%})")
    logger.info(f"  [Filter/{exp_name}] B={filter_stats['B']}, C={filter_stats['C']}, "
                f"A={filter_stats['A']}, D={filter_stats['D']}, "
                f"E={filter_stats['E']}, overflow={filter_stats['overflow']}")

    save_filtered_dataset(filtered_ds, filter_stats, FILTER_SAVE_DIR, exp_name)

    # ── Phase 2：4 种 Loss，各从同一 Phase 1 checkpoint 出发 ─────────────────
    for loss_name in PHASE2_LOSSES:
        logger.info(f"\n  --- Phase 2 / {exp_name} / {loss_name} ---")
        try:
            summary = run_phase2(
                name           = exp_name,
                loss_name      = loss_name,
                model_path     = p1_model_path,
                filtered_ds    = filtered_ds,
                eval_orig_ds   = eval_orig,
                eval_hard_ds   = eval_hard,
                p1_results     = p1_results,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.error(f"  Phase2/{exp_name}/{loss_name} FAILED: {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. 最终汇总
# ─────────────────────────────────────────────────────────────────────────────
summary_path = Path(f"models/{RUN_ID}_compare/logs/summary_all_{RUN_ID}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, ensure_ascii=False, indent=2)

logger.info("\n" + "=" * 70)
logger.info("ALL EXPERIMENTS COMPLETE")
logger.info("=" * 70)
logger.info(f"{'Exp':<12} {'Loss':<28} {'Train':>7} "
            f"{'P1-Orig':>8} {'P2-Orig':>8} {'ΔOrig':>7} "
            f"{'P1-Hard':>8} {'P2-Hard':>8} {'ΔHard':>7}")
logger.info("-" * 90)
for s in all_summaries:
    logger.info(
        f"{s['exp_name']:<12} {s['loss_name']:<28} {s['train_size']:>7,} "
        f"{s['p1_orig']:>8.4f} {s['p2_orig']:>8.4f} {s['delta_p2_orig']:>+7.4f} "
        f"{s['p1_hard']:>8.4f} {s['p2_hard']:>8.4f} {s['delta_p2_hard']:>+7.4f}"
    )
logger.info("=" * 70)
logger.info(f"Summary → {summary_path}")
