#!/usr/bin/env python3
"""
evaluate/run_eval.py
====================
对多个模型做全面评估，指标与 MTEB 排行榜完全一致。

MTEB 对齐说明
─────────────
· 基准：MTEB(eng, v1)，STS 子集共 10 个任务：
    BIOSSES, SICK-R, STS12, STS13, STS14, STS15, STS16, STS17, STS22, STSBenchmark
· 主指标：cosine_spearman（= main_score），通过 result.get_score() 提取
· 所有任务使用 test split
· 平均分 = 10 个任务 cosine_spearman 的算术平均

Part 1: MTEB STS 评估（cosine_spearman，与排行榜一致）
Part 2: 自建 hard-neg 数据集 TripletEvaluator 评估（cosine accuracy）

用法示例：
  python evaluate/run_eval.py \\
      --nli      models/nli_run/final \\
      --llm      models/20260420_compare/llm/final \\
      --regular  models/20260420_compare/regular/final \\
      --combined models/20260420_compare/combined/final

  # 跳过 MTEB（只跑 hard-neg 自评）
  python evaluate/run_eval.py --nli ... --llm ... --regular ... --combined ... --skip_mteb

  # 调试模式（只跑 STSBenchmark，hard-neg 前 100 条）
  python evaluate/run_eval.py --nli ... --llm ... --regular ... --combined ... --debug

输出：
  evaluate/results/{timestamp}/
    summary.txt          # 纯文本汇总表
    report.md            # Markdown 报告
    charts/
      mteb_comparison.png
      hardneg_comparison.png
      overview.png
    raw/
      mteb_{model_key}.json   # 每个模型的 per-task 分数
      hardneg_results.json
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

# MTEB(eng, v1) STS 任务集（10 个），与排行榜一致
# 来源：mteb.get_benchmark('MTEB(eng, v1)') 中 type='STS' 的子集
MTEB_STS_TASKS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]

# 绘图色板
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
def run_mteb_eval(
    model_key: str,
    model_path: str,
    tasks: list[str],
    raw_dir: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    对单个模型运行 MTEB STS 评估。
    主指标：cosine_spearman（= main_score），与排行榜一致。
    返回 {task_name: cosine_spearman}，失败任务值为 nan。
    """
    import mteb
    from sentence_transformers import SentenceTransformer

    logger.info(f"  [MTEB] Loading model: {model_path}")
    st_model = SentenceTransformer(model_path)

    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    evaluation = mteb.MTEB(tasks=task_objects)

    out_dir = raw_dir / f"mteb_{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [MTEB] Running {len(tasks)} tasks ...")
    # encode_kwargs 不传 normalize_embeddings，使用模型默认行为，
    # 确保与 MTEB 排行榜评测方式一致
    results = evaluation.run(
        st_model,
        output_folder=str(out_dir),
        verbosity=0,
        encode_kwargs={"batch_size": 64},
    )

    scores: dict[str, float] = {}
    raw: dict[str, dict] = {}

    for result in results:
        name = result.task_name
        # result.get_score() 返回 main_score 的均值 = cosine_spearman
        # 这与 MTEB 排行榜计算方式完全一致
        try:
            score = float(result.get_score())
        except Exception:
            score = float("nan")

        scores[name] = score
        raw[name] = {
            "cosine_spearman": score,
            "scores": result.scores,
        }

        if not np.isnan(score):
            logger.info(f"    {name}: {score:.4f}")
        else:
            logger.warning(f"    {name}: FAILED (nan)")

    avg = _sts_avg(scores)
    logger.info(f"  [MTEB] Average cosine_spearman ({len(scores)} tasks): {avg:.4f}")

    (raw_dir / f"mteb_{model_key}.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scores


def _sts_avg(scores: dict[str, float]) -> float:
    """10 个任务的算术平均（忽略 nan）。"""
    vals = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Hard-neg 数据集评估
# ─────────────────────────────────────────────────────────────────────────────
def load_hardneg_data(
    paths: list[str],
    max_samples: int | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
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

    return (
        [r["anchor"]   for r in rows],
        [r["pos"]      for r in rows],
        [r["neg"]      for r in rows],
        [r["hard_neg"] for r in rows],
    )


def run_hardneg_eval(
    model_key: str,
    model_path: str,
    anchors: list[str],
    positives: list[str],
    neg_orig: list[str],
    neg_hard: list[str],
    logger: logging.Logger,
) -> dict[str, float]:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import TripletEvaluator

    logger.info(f"  [Hard-neg] Loading model: {model_path}")
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

    logger.info(f"    orig neg accuracy : {orig_acc:.4f}")
    logger.info(f"    hard neg accuracy : {hard_acc:.4f}")
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
    n_tasks = len(task_names)
    n_models = len(model_keys)
    x = np.arange(n_tasks)
    width = 0.72 / n_models

    for i, key in enumerate(model_keys):
        scores = [scores_by_model[key].get(t, float("nan")) for t in task_names]
        bars = ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            scores, width,
            label=MODEL_LABELS.get(key, key),
            color=COLORS.get(key, "#333333"),
            alpha=0.88, edgecolor="white", linewidth=0.6,
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
    model_keys = list(mteb_scores.keys())
    all_tasks = sorted({t for m in mteb_scores.values() for t in m})

    # 末尾加 Average 列
    avg_task = "Average"
    scores_with_avg: dict[str, dict[str, float]] = {}
    for k in model_keys:
        d = dict(mteb_scores[k])
        d[avg_task] = _sts_avg(mteb_scores[k])
        scores_with_avg[k] = d
    display_tasks = all_tasks + [avg_task]

    plt.rcParams.update(_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    _bar_group(ax, display_tasks, model_keys, scores_with_avg,
               title="MTEB(eng, v1) STS Tasks — cosine_spearman",
               ylabel="cosine_spearman")
    # 在 Average 列前画分隔线
    ax.axvline(x=len(all_tasks) - 0.5, color="#AAAAAA", linewidth=1.0, linestyle="--")
    fig.suptitle("MTEB STS Benchmark Comparison  [MTEB(eng, v1)]",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hardneg(hardneg_scores: dict[str, dict[str, float]], out_path: Path):
    model_keys = list(hardneg_scores.keys())
    categories = ["Original Neg", "Hard Neg"]
    scores_by_model = {
        k: {
            "Original Neg": hardneg_scores[k].get("orig_acc", float("nan")),
            "Hard Neg":     hardneg_scores[k].get("hard_acc", float("nan")),
        }
        for k in model_keys
    }

    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    _bar_group(axes[0], categories, model_keys, scores_by_model,
               title="Triplet Accuracy by Negative Type",
               ylabel="Cosine Accuracy", ylim=(0.0, 1.1))

    x = np.arange(len(model_keys))
    width = 0.35
    orig_vals = [hardneg_scores[k].get("orig_acc", float("nan")) for k in model_keys]
    hard_vals = [hardneg_scores[k].get("hard_acc", float("nan")) for k in model_keys]

    axes[1].bar(x - width / 2, orig_vals, width, label="Orig Neg",
                color="#888780", alpha=0.85, edgecolor="white")
    axes[1].bar(x + width / 2, hard_vals, width, label="Hard Neg",
                color="#D94F3D", alpha=0.85, edgecolor="white")
    for xi, (o, h) in enumerate(zip(orig_vals, hard_vals)):
        if not (np.isnan(o) or np.isnan(h)):
            delta = h - o
            axes[1].annotate(
                f"Δ={delta:+.3f}",
                xy=(xi, max(o, h) + 0.02),
                ha="center", fontsize=8,
                color="#D94F3D" if delta < 0 else "#1D9E75",
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys],
        rotation=12, ha="right",
    )
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
    model_keys = list(mteb_scores.keys()) or list(hardneg_scores.keys())
    mteb_avg = {k: _sts_avg(mteb_scores.get(k, {})) for k in model_keys}

    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("MTEB STS Avg\n(cosine_spearman, 10 tasks)",
         {k: mteb_avg.get(k, float("nan")) for k in model_keys}),
        ("Hard-Neg: Orig Neg\n(Cosine Accuracy)",
         {k: hardneg_scores.get(k, {}).get("orig_acc", float("nan")) for k in model_keys}),
        ("Hard-Neg: Hard Neg\n(Cosine Accuracy)",
         {k: hardneg_scores.get(k, {}).get("hard_acc", float("nan")) for k in model_keys}),
    ]

    for ax, (title, score_dict) in zip(axes, metrics):
        vals  = [score_dict.get(k, float("nan")) for k in model_keys]
        clrs  = [COLORS.get(k, "#333333") for k in model_keys]
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
        ax.set_xticklabels(
            [MODEL_LABELS.get(k, k) for k in model_keys],
            rotation=12, ha="right", fontsize=8.5,
        )
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
        "=" * 76,
        "Model Evaluation Summary",
        f"Date       : {timestamp}",
        "Benchmark  : MTEB(eng, v1) STS — 10 tasks, metric: cosine_spearman",
        "=" * 76,
        "",
        "Models",
        "-" * 76,
    ]
    for k, p in model_paths.items():
        lines.append(f"  {k:<12}: {p}")

    if mteb_scores:
        all_tasks = sorted({t for m in mteb_scores.values() for t in m})
        col_w = 14
        header = f"  {'Task':<24}" + "".join(f"  {k:<{col_w}}" for k in model_paths)
        lines += ["", "MTEB STS Results (cosine_spearman)", "-" * 76, header,
                  "  " + "-" * (len(header) - 2)]
        for t in all_tasks:
            row = f"  {t:<24}"
            for k in model_paths:
                v = mteb_scores.get(k, {}).get(t, float("nan"))
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
            lines.append(row)
        lines.append("  " + "-" * (len(header) - 2))
        avg_row = f"  {'Average (10 tasks)':<24}"
        for k in model_paths:
            avg = _sts_avg(mteb_scores.get(k, {}))
            avg_row += f"  {avg:>{col_w}.4f}" if not np.isnan(avg) else f"  {'N/A':>{col_w}}"
        lines.append(avg_row)

    if hardneg_scores:
        col_w = 14
        header2 = f"  {'Metric':<30}" + "".join(f"  {k:<{col_w}}" for k in model_paths)
        lines += [
            "",
            f"Hard-Neg Dataset Results (TripletEvaluator cosine accuracy, n={n_hardneg_samples:,})",
            "-" * 76, header2, "  " + "-" * (len(header2) - 2),
        ]
        for metric, label in [("orig_acc", "Orig Neg Accuracy"),
                               ("hard_acc", "Hard Neg Accuracy")]:
            row = f"  {label:<30}"
            for k in model_paths:
                v = hardneg_scores.get(k, {}).get(metric, float("nan"))
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
            lines.append(row)

    lines += ["", "=" * 76]
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
        f"**Date**: {timestamp}  ",
        "**Benchmark**: MTEB(eng, v1) STS — 10 tasks, metric: `cosine_spearman`",
        "",
        "## Models",
        "",
        "| Key | Path |",
        "|-----|------|",
    ]
    for k, p in model_paths.items():
        lines.append(f"| `{k}` | `{p}` |")

    if mteb_scores:
        all_tasks = sorted({t for m in mteb_scores.values() for t in m})
        header_cols = " | ".join(MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys)
        sep_cols = "|".join("------" for _ in model_keys)
        lines += [
            "",
            "## Part 1: MTEB STS Benchmark",
            "",
            "> Metric: **cosine_spearman**  ·  Benchmark: **MTEB(eng, v1)**  ·  10 tasks",
            "",
            "![MTEB Comparison](charts/mteb_comparison.png)",
            "",
            f"| Task | {header_cols} |",
            f"|------|{sep_cols}|",
        ]
        for t in all_tasks:
            row = f"| {t} |"
            for k in model_keys:
                row += " " + fmt(mteb_scores.get(k, {}).get(t, float("nan"))) + " |"
            lines.append(row)
        avg_row = "| **Average** |"
        for k in model_keys:
            avg_row += " **" + fmt(_sts_avg(mteb_scores.get(k, {}))) + "** |"
        lines.append(avg_row)

    if hardneg_scores:
        header_cols = " | ".join(MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys)
        sep_cols = "|".join("------" for _ in model_keys)
        lines += [
            "",
            "## Part 2: Hard-Neg Dataset Evaluation",
            "",
            f"> Metric: **Cosine Accuracy** (TripletEvaluator, n={n_hardneg_samples:,})",
            "",
            "![Hard-Neg Comparison](charts/hardneg_comparison.png)",
            "",
            f"| Metric | {header_cols} |",
            f"|--------|{sep_cols}|",
        ]
        for metric, label in [("orig_acc", "Original Neg Accuracy"),
                               ("hard_acc", "Hard Neg Accuracy")]:
            row = f"| {label} |"
            for k in model_keys:
                row += " " + fmt(hardneg_scores.get(k, {}).get(metric, float("nan"))) + " |"
            lines.append(row)

    lines += ["", "## Overview", "", "![Overview](charts/overview.png)"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on MTEB(eng, v1) STS + hard-neg dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base",     default=DEFAULT_BASE_MODEL)
    parser.add_argument("--nli",      required=True)
    parser.add_argument("--llm",      required=True)
    parser.add_argument("--regular",  required=True)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--hn_llm",      nargs="+", default=DEFAULT_HARD_NEG_DATA["llm"])
    parser.add_argument("--hn_regular",  nargs="+", default=DEFAULT_HARD_NEG_DATA["regular"])
    parser.add_argument("--hn_combined", nargs="+", default=DEFAULT_HARD_NEG_DATA["combined"])
    parser.add_argument("--tasks", nargs="+", default=MTEB_STS_TASKS,
                        help="MTEB STS 任务列表（默认 MTEB(eng,v1) 的 10 个 STS 任务）")
    parser.add_argument("--skip_mteb",    action="store_true")
    parser.add_argument("--skip_hardneg", action="store_true")
    parser.add_argument("--max_hardneg_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：只跑 STSBenchmark，hard-neg 前 100 条")
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

    if args.debug:
        args.tasks = ["STSBenchmark"]
        args.max_hardneg_samples = 100
        logger.info("[DEBUG] tasks=STSBenchmark only, max_hardneg_samples=100")

    model_paths = {
        "base":     args.base,
        "nli":      args.nli,
        "llm":      args.llm,
        "regular":  args.regular,
        "combined": args.combined,
    }
    logger.info(f"Tasks   : {args.tasks}")
    logger.info(f"Models  : { {k: v for k, v in model_paths.items()} }")

    mteb_scores:    dict[str, dict[str, float]] = {}
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
                logger.error(f"  MTEB failed for {key}: {exc}", exc_info=True)
                mteb_scores[key] = {}

        (raw_dir / "mteb_all.json").write_text(
            json.dumps(mteb_scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 打印汇总表
        logger.info("\n" + "=" * 60)
        logger.info("MTEB STS Summary (cosine_spearman)")
        logger.info("=" * 60)
        col_w = 10
        header = f"  {'Task':<22}" + "".join(f"  {k:<{col_w}}" for k in model_paths)
        logger.info(header)
        all_tasks = sorted({t for m in mteb_scores.values() for t in m})
        for t in all_tasks:
            row = f"  {t:<22}"
            for k in model_paths:
                v = mteb_scores.get(k, {}).get(t, float("nan"))
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
            logger.info(row)
        logger.info("  " + "-" * (len(header) - 2))
        avg_row = f"  {'Average':<22}"
        for k in model_paths:
            avg = _sts_avg(mteb_scores.get(k, {}))
            avg_row += f"  {avg:>{col_w}.4f}" if not np.isnan(avg) else f"  {'N/A':>{col_w}}"
        logger.info(avg_row)
    else:
        logger.info("[Part 1] MTEB skipped.")

    # ── Part 2: Hard-neg 评估 ──────────────────────────────────────────────
    if not args.skip_hardneg:
        logger.info("\n[Part 2] Hard-Neg Dataset Evaluation")
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
                logger.error(f"  Hard-neg eval failed for {key}: {exc}", exc_info=True)
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
            logger.error(f"  mteb plot failed: {exc}")
    if hardneg_scores:
        try:
            plot_hardneg(hardneg_scores, charts_dir / "hardneg_comparison.png")
            logger.info("  → hardneg_comparison.png")
        except Exception as exc:
            logger.error(f"  hardneg plot failed: {exc}")
    if mteb_scores or hardneg_scores:
        try:
            plot_overview(mteb_scores, hardneg_scores, charts_dir / "overview.png")
            logger.info("  → overview.png")
        except Exception as exc:
            logger.error(f"  overview plot failed: {exc}")

    # ── 报告 ────────────────────────────────────────────────────────────────
    logger.info("\n[Reports]")
    try:
        write_summary_txt(model_paths, mteb_scores, hardneg_scores,
                          n_hardneg_samples, out_base / "summary.txt", timestamp)
        logger.info(f"  → summary.txt")
    except Exception as exc:
        logger.error(f"  summary.txt failed: {exc}")
    try:
        write_report_md(model_paths, mteb_scores, hardneg_scores,
                        n_hardneg_samples, out_base / "report.md", timestamp)
        logger.info(f"  → report.md")
    except Exception as exc:
        logger.error(f"  report.md failed: {exc}")

    logger.info(f"\nEvaluation complete. Results: {out_base}")


if __name__ == "__main__":
    main()
