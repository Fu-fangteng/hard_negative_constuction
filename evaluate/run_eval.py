#!/usr/bin/env python3
"""
evaluate/run_eval.py
====================
对三个模型做全面评估。

  base     - all-MiniLM-L6-v2 原始预训练权重（HuggingFace）
  nli      - 用 nli-for-simcse 微调的 all-MiniLM-L6-v2
  hard_neg - 用我们构造的 hard-neg 数据微调的 all-MiniLM-L6-v2

Part 1: MTEB STS 评估（spearman cosine）
Part 2: 自建 hard-neg 数据集 TripletEvaluator 评估（cosine accuracy）

用法示例：
  python evaluate/run_eval.py \\
      --nli    models/nli_run/final \\
      --hard_neg models/20260420_123456_compare/combined/final

输出：
  evaluate/results/{timestamp}/
    summary.txt
    report.md
    charts/mteb_comparison.png
    charts/hardneg_comparison.png
    charts/overview.png
    raw/mteb_base.json  raw/mteb_nli.json  raw/mteb_hard_neg.json
    raw/hardneg_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "all-MiniLM-L6-v2"

DEFAULT_HARD_NEG_DATA = {
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

# MTEB 标准 STS 英文任务
MTEB_STS_TASKS = [
    "STS12", "STS13", "STS14", "STS15", "STS16",
    "STSBenchmark", "SICK-R",
]

# 绘图色板（5个模型）
COLORS = {
    "base":     "#888780",
    "nli":      "#378ADD",
    "llm":      "#9B59B6",
    "regular":  "#1D9E75",
    "combined": "#D94F3D",
}
MODEL_LABELS = {
    "base":     "Base",
    "nli":      "NLI FT",
    "llm":      "Hard-Neg\n(LLM)",
    "regular":  "Hard-Neg\n(Regular)",
    "combined": "Hard-Neg\n(Combined)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: MTEB 评估
# ─────────────────────────────────────────────────────────────────────────────
def _extract_mteb_score(result) -> float:
    """从 MTEB TaskResult 中提取主指标（spearman cosine）。兼容 MTEB v1/v2。"""
    # 方法 A：新版 MTEB 的 get_score()
    if hasattr(result, "get_score"):
        try:
            return float(result.get_score())
        except Exception:
            pass

    # 方法 B：解析 .scores dict
    if hasattr(result, "scores") and result.scores:
        for split in ("test", "validation", "dev"):
            entries = result.scores.get(split, [])
            if not entries:
                continue
            s = entries[0] if isinstance(entries, list) else entries
            for key in ("spearman_cosine", "cos_sim_spearman", "main_score"):
                if key in s:
                    return float(s[key])

    return float("nan")


def run_mteb_eval(
    model_key: str,
    model_path: str,
    tasks: list[str],
    raw_dir: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    对单个模型运行 MTEB STS 评估。
    返回 {task_name: spearman_cosine}，失败任务值为 nan。
    """
    import mteb
    from sentence_transformers import SentenceTransformer

    logger.info(f"  [MTEB] Loading: {model_path}")
    st_model = SentenceTransformer(model_path)

    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    benchmark = mteb.MTEB(tasks=task_objects)

    out_dir = raw_dir / f"mteb_{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [MTEB] Running {len(tasks)} tasks ...")
    results = benchmark.run(st_model, output_folder=str(out_dir), verbosity=0)

    scores: dict[str, float] = {}
    raw: dict[str, dict] = {}
    for r in results:
        name = r.task_name
        score = _extract_mteb_score(r)
        scores[name] = score
        raw[name] = {"score": score, "scores": getattr(r, "scores", {})}
        logger.info(f"    {name}: {score:.4f}" if not np.isnan(score) else f"    {name}: N/A")

    # 保存原始 JSON
    (raw_dir / f"mteb_{model_key}.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Hard-neg 数据集评估
# ─────────────────────────────────────────────────────────────────────────────
def load_hardneg_data(
    paths: list[str],
    max_samples: int | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    加载 constructed_data.json，过滤 success=True，去重。
    返回 (anchors, positives, neg_orig, neg_hard)。
    """
    seen: set[str] = set()
    rows: list[dict] = []
    for path in paths:
        p = PROJECT_ROOT / path
        assert p.exists(), f"找不到数据文件: {p}"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if not item.get("success") or not item.get("hard_neg"):
                continue
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            rows.append(item)

    if max_samples:
        rows = rows[:max_samples]

    anchors   = [r["anchor"]   for r in rows]
    positives = [r["pos"]      for r in rows]
    neg_orig  = [r["neg"]      for r in rows]
    neg_hard  = [r["hard_neg"] for r in rows]
    return anchors, positives, neg_orig, neg_hard


def run_hardneg_eval(
    model_key: str,
    model_path: str,
    anchors: list[str],
    positives: list[str],
    neg_orig: list[str],
    neg_hard: list[str],
    logger: logging.Logger,
) -> dict[str, float]:
    """
    用 TripletEvaluator 评估模型。
    返回 {orig_acc, hard_acc}（cosine accuracy）。
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import TripletEvaluator

    logger.info(f"  [Hard-neg] Loading: {model_path}")
    model = SentenceTransformer(model_path)

    def _acc(evaluator) -> float:
        res = evaluator(model)
        for k, v in res.items():
            if "accuracy" in k.lower():
                return float(v)
        return float("nan")

    orig_acc = _acc(TripletEvaluator(
        anchors=anchors, positives=positives, negatives=neg_orig,
        name=f"{model_key}-orig",
    ))
    hard_acc = _acc(TripletEvaluator(
        anchors=anchors, positives=positives, negatives=neg_hard,
        name=f"{model_key}-hard",
    ))

    logger.info(f"    orig neg accuracy: {orig_acc:.4f}")
    logger.info(f"    hard neg accuracy: {hard_acc:.4f}")
    return {"orig_acc": orig_acc, "hard_acc": hard_acc}


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────
_PLOT_STYLE = {
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#888780", "axes.grid": True,
    "grid.color": "#EBEBEB", "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
}


def _bar_group(ax, task_names, model_keys, scores_by_model, title, ylabel,
               ylim=(0.0, 1.0), show_legend=True):
    """通用分组柱状图绘制。scores_by_model: {model_key: {task: value}}"""
    n_tasks = len(task_names)
    n_models = len(model_keys)
    x = np.arange(n_tasks)
    width = 0.72 / n_models

    for i, key in enumerate(model_keys):
        scores = [scores_by_model[key].get(t, float("nan")) for t in task_names]
        bars = ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            scores,
            width,
            label=MODEL_LABELS[key],
            color=COLORS[key],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, scores):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")


def plot_mteb(mteb_scores: dict[str, dict[str, float]], out_path: Path):
    """MTEB STS 各任务分组柱状图。"""
    model_keys = list(mteb_scores.keys())
    # 取所有模型都有结果的任务
    all_tasks = sorted({t for m in mteb_scores.values() for t in m})

    plt.rcParams.update(_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    _bar_group(ax, all_tasks, model_keys, mteb_scores,
               title="MTEB STS Tasks — Spearman Cosine",
               ylabel="Spearman Cosine Similarity")
    fig.suptitle("MTEB STS Benchmark Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hardneg(hardneg_scores: dict[str, dict[str, float]], out_path: Path):
    """Hard-neg TripletEvaluator 柱状图（orig neg vs hard neg）。"""
    model_keys = list(hardneg_scores.keys())
    categories = ["Original Neg", "Hard Neg"]

    # _bar_group 期望 {model_key: {category: value}}，不能反转
    scores_by_model: dict[str, dict[str, float]] = {
        k: {
            "Original Neg": hardneg_scores[k].get("orig_acc", float("nan")),
            "Hard Neg":     hardneg_scores[k].get("hard_acc", float("nan")),
        }
        for k in model_keys
    }

    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：按负样本类型分组（每类别下5个模型的柱子）
    _bar_group(axes[0], categories, model_keys, scores_by_model,
               title="Triplet Accuracy by Negative Type",
               ylabel="Cosine Accuracy",
               ylim=(0.0, 1.1))

    # 右图：按模型分组（Δ between orig and hard）
    x = np.arange(len(model_keys))
    width = 0.35
    orig_vals = [hardneg_scores[k]["orig_acc"] for k in model_keys]
    hard_vals = [hardneg_scores[k]["hard_acc"] for k in model_keys]

    axes[1].bar(x - width / 2, orig_vals, width, label="Orig Neg",
                color="#888780", alpha=0.85, edgecolor="white")
    axes[1].bar(x + width / 2, hard_vals, width, label="Hard Neg",
                color="#D94F3D", alpha=0.85, edgecolor="white")
    for xi, (o, h) in enumerate(zip(orig_vals, hard_vals)):
        delta = h - o
        axes[1].annotate(
            f"Δ={delta:+.3f}",
            xy=(xi, max(o, h) + 0.02),
            ha="center", fontsize=8,
            color="#D94F3D" if delta < 0 else "#1D9E75",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[k].replace("\n", " ") for k in model_keys],
                             rotation=12, ha="right")
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_title("Orig vs Hard Neg Accuracy per Model")
    axes[1].set_ylabel("Cosine Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, color="#EBEBEB", linewidth=0.8)

    fig.suptitle("Hard-Neg Dataset Evaluation", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    mteb_scores: dict[str, dict[str, float]],
    hardneg_scores: dict[str, dict[str, float]],
    out_path: Path,
):
    """综合总览图：MTEB 平均分 + Hard-neg 两项指标。"""
    model_keys = list(mteb_scores.keys()) or list(hardneg_scores.keys())

    # MTEB 平均分（忽略 nan）
    mteb_avg = {}
    for k in model_keys:
        vals = [v for v in mteb_scores.get(k, {}).values() if not np.isnan(v)]
        mteb_avg[k] = np.mean(vals) if vals else float("nan")

    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("MTEB STS Avg\n(Spearman Cosine)",
         {k: mteb_avg.get(k, float("nan")) for k in model_keys}),
        ("Hard-Neg: Orig Neg\n(Cosine Accuracy)",
         {k: hardneg_scores.get(k, {}).get("orig_acc", float("nan")) for k in model_keys}),
        ("Hard-Neg: Hard Neg\n(Cosine Accuracy)",
         {k: hardneg_scores.get(k, {}).get("hard_acc", float("nan")) for k in model_keys}),
    ]

    for ax, (title, score_dict) in zip(axes, metrics):
        vals  = [score_dict.get(k, float("nan")) for k in model_keys]
        clrs  = [COLORS[k] for k in model_keys]
        bars = ax.bar(range(len(model_keys)), vals, color=clrs,
                      edgecolor="white", linewidth=0.8, alpha=0.88)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
        ax.set_xticks(range(len(model_keys)))
        ax.set_xticklabels([MODEL_LABELS[k] for k in model_keys],
                           rotation=12, ha="right", fontsize=8.5)
        ax.set_ylim(0.0, 1.1)
        ax.set_title(title)
        ax.set_ylabel("Score")

    fig.suptitle("Model Comparison Overview", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_txt(
    model_paths: dict[str, str],
    mteb_scores: dict[str, dict[str, float]],
    hardneg_scores: dict[str, dict[str, float]],
    n_hardneg_samples: int,
    out_path: Path,
    timestamp: str,
):
    lines = [
        "=" * 70,
        "Model Evaluation Summary",
        f"Date      : {timestamp}",
        "=" * 70,
        "",
        "Models",
        "-" * 70,
    ]
    for k, p in model_paths.items():
        lines.append(f"  {k:<10}: {p}")

    # MTEB
    if mteb_scores:
        lines += ["", "MTEB STS Results (Spearman Cosine)", "-" * 70]
        all_tasks = sorted({t for m in mteb_scores.values() for t in m})
        header = f"  {'Task':<20}" + "".join(f"  {k:<14}" for k in model_paths)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for t in all_tasks:
            row = f"  {t:<20}"
            for k in model_paths:
                v = mteb_scores.get(k, {}).get(t, float("nan"))
                row += f"  {v:>14.4f}" if not np.isnan(v) else f"  {'N/A':>14}"
            lines.append(row)
        # Averages
        lines.append("  " + "-" * (len(header) - 2))
        avg_row = f"  {'Average':<20}"
        for k in model_paths:
            vals = [v for v in mteb_scores.get(k, {}).values() if not np.isnan(v)]
            avg = np.mean(vals) if vals else float("nan")
            avg_row += f"  {avg:>14.4f}" if not np.isnan(avg) else f"  {'N/A':>14}"
        lines.append(avg_row)

    # Hard-neg
    if hardneg_scores:
        lines += [
            "",
            f"Hard-Neg Dataset Results (TripletEvaluator, n={n_hardneg_samples:,})",
            "-" * 70,
            f"  {'Metric':<25}" + "".join(f"  {k:<14}" for k in model_paths),
            "  " + "-" * 60,
        ]
        for metric, label in [("orig_acc", "Orig Neg Accuracy"),
                               ("hard_acc", "Hard Neg Accuracy")]:
            row = f"  {label:<25}"
            for k in model_paths:
                v = hardneg_scores.get(k, {}).get(metric, float("nan"))
                row += f"  {v:>14.4f}" if not np.isnan(v) else f"  {'N/A':>14}"
            lines.append(row)

    lines += ["", "=" * 70]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_md(
    model_paths: dict[str, str],
    mteb_scores: dict[str, dict[str, float]],
    hardneg_scores: dict[str, dict[str, float]],
    n_hardneg_samples: int,
    out_path: Path,
    timestamp: str,
):
    model_keys = list(model_paths.keys())

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    lines = [
        "# Model Evaluation Report",
        "",
        f"**Date**: {timestamp}",
        "",
        "## Models",
        "",
        "| Key | Path |",
        "|-----|------|",
    ]
    for k, p in model_paths.items():
        lines.append(f"| `{k}` | `{p}` |")

    # MTEB table
    if mteb_scores:
        all_tasks = sorted({t for m in mteb_scores.values() for t in m})
        lines += [
            "",
            "## Part 1: MTEB STS Benchmark",
            "",
            "Metric: **Spearman Cosine Similarity**",
            "",
            "![MTEB Comparison](charts/mteb_comparison.png)",
            "",
            "| Task | " + " | ".join(MODEL_LABELS[k].replace("\n", " ") for k in model_keys) + " |",
            "|------|" + "|".join("------" for _ in model_keys) + "|",
        ]
        for t in all_tasks:
            row = f"| {t} |"
            for k in model_keys:
                row += " " + fmt(mteb_scores.get(k, {}).get(t, float("nan"))) + " |"
            lines.append(row)
        # Average row
        avg_row = "| **Average** |"
        for k in model_keys:
            vals = [v for v in mteb_scores.get(k, {}).values() if not np.isnan(v)]
            avg_row += " **" + fmt(np.mean(vals) if vals else float("nan")) + "** |"
        lines.append(avg_row)

    # Hard-neg table
    if hardneg_scores:
        lines += [
            "",
            "## Part 2: Hard-Neg Dataset Evaluation",
            "",
            f"Metric: **Cosine Accuracy** (TripletEvaluator, n={n_hardneg_samples:,})",
            "",
            "![Hard-Neg Comparison](charts/hardneg_comparison.png)",
            "",
            "| Metric | " + " | ".join(MODEL_LABELS[k].replace("\n", " ") for k in model_keys) + " |",
            "|--------|" + "|".join("------" for _ in model_keys) + "|",
        ]
        for metric, label in [("orig_acc", "Original Neg Accuracy"),
                               ("hard_acc", "Hard Neg Accuracy")]:
            row = f"| {label} |"
            for k in model_keys:
                row += " " + fmt(hardneg_scores.get(k, {}).get(metric, float("nan"))) + " |"
            lines.append(row)

    lines += [
        "",
        "## Overview",
        "",
        "![Overview](charts/overview.png)",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 5 models on MTEB STS + hard-neg dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 5 个模型路径 ──────────────────────────────────────────────────────
    parser.add_argument("--base",     default=DEFAULT_BASE_MODEL,
                        help="Base 模型路径或 HF model ID")
    parser.add_argument("--nli",      required=True,
                        help="NLI 微调模型路径或 HF model ID")
    parser.add_argument("--llm",      required=True,
                        help="Hard-neg(LLM) 微调模型路径")
    parser.add_argument("--regular",  required=True,
                        help="Hard-neg(Regular) 微调模型路径")
    parser.add_argument("--combined", required=True,
                        help="Hard-neg(Combined) 微调模型路径")
    # ── Hard-neg 数据路径（可覆盖默认）────────────────────────────────────
    parser.add_argument("--hn_llm",      nargs="+",
                        default=DEFAULT_HARD_NEG_DATA["llm"],
                        help="LLM constructed_data.json 路径")
    parser.add_argument("--hn_regular",  nargs="+",
                        default=DEFAULT_HARD_NEG_DATA["regular"],
                        help="Regular constructed_data.json 路径")
    parser.add_argument("--hn_combined", nargs="+",
                        default=DEFAULT_HARD_NEG_DATA["combined"],
                        help="Combined constructed_data.json 路径（LLM+Regular）")
    # ── 评估控制 ──────────────────────────────────────────────────────────
    parser.add_argument("--tasks", nargs="+", default=MTEB_STS_TASKS,
                        help="MTEB 任务名称列表")
    parser.add_argument("--skip_mteb",    action="store_true",
                        help="跳过 MTEB 评估")
    parser.add_argument("--skip_hardneg", action="store_true",
                        help="跳过 hard-neg 数据集评估")
    parser.add_argument("--max_hardneg_samples", type=int, default=None,
                        help="每个 hard-neg 数据集最多使用多少条（默认全部）")
    parser.add_argument("--output_dir", default=None,
                        help="输出目录（默认 evaluate/results/{timestamp}）")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：MTEB 只跑 STSBenchmark，hard-neg 只用前100条")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_dir) if args.output_dir else \
               Path(__file__).parent / "results" / timestamp
    charts_dir = out_base / "charts"
    raw_dir    = out_base / "raw"
    for d in [out_base, charts_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(out_base / f"eval_{timestamp}.log")
    logger.info(f"Output: {out_base}")

    # 调试模式覆盖
    if args.debug:
        args.tasks = ["STSBenchmark"]
        args.max_hardneg_samples = 100
        logger.info("[DEBUG] tasks=STSBenchmark, max_hardneg_samples=100")

    model_paths = {
        "base":     args.base,
        "nli":      args.nli,
        "llm":      args.llm,
        "regular":  args.regular,
        "combined": args.combined,
    }
    # hard-neg 评估数据集（各模型对应自己的构造数据）
    hardneg_data_paths = {
        "llm":      args.hn_llm,
        "regular":  args.hn_regular,
        "combined": args.hn_combined,
    }
    logger.info(f"Models: { {k: v for k, v in model_paths.items()} }")

    mteb_scores:    dict[str, dict[str, float]] = {}
    # hardneg_scores: {model_key: {dataset_key: {orig_acc, hard_acc}}}
    # 为便于绘图，统一用 combined 数据集评估所有模型（公平对比）
    # 同时也保存各自数据集上的评估结果
    hardneg_scores: dict[str, dict[str, float]] = {}
    n_hardneg_samples = 0

    # ── Part 1: MTEB ──────────────────────────────────────────────────────
    if not args.skip_mteb:
        logger.info("\n[Part 1] MTEB STS Evaluation")
        for key, path in model_paths.items():
            logger.info(f"\n  Model: {key}")
            try:
                mteb_scores[key] = run_mteb_eval(key, path, args.tasks, raw_dir, logger)
            except Exception as exc:
                logger.error(f"  MTEB failed for {key}: {exc}")
                mteb_scores[key] = {}

        (raw_dir / "mteb_all.json").write_text(
            json.dumps(mteb_scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        logger.info("[Part 1] MTEB skipped.")

    # ── Part 2: Hard-neg 数据集（用 combined 数据集对所有模型做公平评估）──
    if not args.skip_hardneg:
        logger.info("\n[Part 2] Hard-Neg Dataset Evaluation (shared combined dataset)")
        anchors, positives, neg_orig, neg_hard = load_hardneg_data(
            args.hn_combined, max_samples=args.max_hardneg_samples
        )
        n_hardneg_samples = len(anchors)
        logger.info(f"  Shared eval set: {n_hardneg_samples:,} samples (combined)")

        for key, path in model_paths.items():
            logger.info(f"\n  Model: {key}")
            try:
                hardneg_scores[key] = run_hardneg_eval(
                    key, path, anchors, positives, neg_orig, neg_hard, logger
                )
            except Exception as exc:
                logger.error(f"  Hard-neg eval failed for {key}: {exc}")
                hardneg_scores[key] = {}

        (raw_dir / "hardneg_results.json").write_text(
            json.dumps(hardneg_scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        logger.info("[Part 2] Hard-neg evaluation skipped.")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    logger.info("\n[Plots] Generating charts ...")
    if mteb_scores:
        try:
            plot_mteb(mteb_scores, charts_dir / "mteb_comparison.png")
            logger.info("  → mteb_comparison.png")
        except Exception as exc:
            logger.error(f"  mteb_comparison.png failed: {exc}")

    if hardneg_scores:
        try:
            plot_hardneg(hardneg_scores, charts_dir / "hardneg_comparison.png")
            logger.info("  → hardneg_comparison.png")
        except Exception as exc:
            logger.error(f"  hardneg_comparison.png failed: {exc}")

    if mteb_scores or hardneg_scores:
        try:
            plot_overview(mteb_scores, hardneg_scores, charts_dir / "overview.png")
            logger.info("  → overview.png")
        except Exception as exc:
            logger.error(f"  overview.png failed: {exc}")

    # ── 报告（无论绘图是否失败，都必须写出）──────────────────────────────────
    logger.info("\n[Reports] Writing summary ...")
    try:
        write_summary_txt(model_paths, mteb_scores, hardneg_scores,
                          n_hardneg_samples, out_base / "summary.txt", timestamp)
        logger.info("  → summary.txt")
    except Exception as exc:
        logger.error(f"  summary.txt failed: {exc}")

    try:
        write_report_md(model_paths, mteb_scores, hardneg_scores,
                        n_hardneg_samples, out_base / "report.md", timestamp)
        logger.info("  → report.md")
    except Exception as exc:
        logger.error(f"  report.md failed: {exc}")

    logger.info(f"\n✅ Evaluation complete. Results: {out_base}")


if __name__ == "__main__":
    main()
