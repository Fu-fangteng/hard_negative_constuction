#!/usr/bin/env python3
"""
evaluate/run_eval.py
====================
在 MTEB(eng, v1) 的 10 个 STS 任务上评测任意模型，生成对比图表与报告。
指标：cosine_spearman，与 MTEB 排行榜完全一致。

用法
────
  python evaluate/run_eval.py --model all-MiniLM-L6-v2
  python evaluate/run_eval.py \
      --model all-MiniLM-L6-v2 \
      --model models/nli_run/final \
      --model models/20260420_compare/combined/phase2_triplet_cascade/final

输出（evaluate/results/{timestamp}/）
  summary.txt          — 纯文本汇总表
  report.md            — Markdown 报告
  charts/scores.png    — 柱状对比图
  raw/scores.json      — 原始 per-task 分数
  eval.log             — 完整运行日志
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"

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

COLORS = ["#888780", "#378ADD", "#1D9E75", "#D94F3D", "#9B59B6",
          "#BA7517", "#E67E22", "#27AE60", "#2980B9", "#8E44AD"]


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估
# ─────────────────────────────────────────────────────────────────────────────
def _extract_score(result, logger: logging.Logger) -> float:
    """从 MTEB TaskResult 中提取 cosine_spearman。"""
    score = float("nan")
    try:
        val = result.get_score()
        if val is not None:
            score = float(val)
    except Exception as exc:
        logger.warning(f"      get_score() failed: {exc}")

    if np.isnan(score):
        try:
            test_s = result.scores.get("test", [])
            if test_s:
                cs = test_s[0].get("cosine_spearman")
                if cs is not None:
                    score = float(cs)
                    logger.info(f"      (fallback) cosine_spearman = {score:.4f}")
        except Exception as exc2:
            logger.warning(f"      fallback failed: {exc2}")
    return score


def run_mteb_eval(
    model_path: str,
    tasks: list[str],
    raw_dir: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    """用官方 mteb.get_model() 加载模型并评测，返回 {task_name: cosine_spearman}。"""
    import mteb

    logger.info(f"  Loading: {model_path}")
    model = mteb.get_model(model_path)

    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    evaluation   = mteb.MTEB(tasks=task_objects)

    out_dir = raw_dir / _short_name(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Running {len(tasks)} MTEB STS tasks ...")
    results = evaluation.run(
        model,
        output_folder=str(out_dir),
        overwrite_results=True,
        eval_splits=["test"],
    )

    scores: dict[str, float] = {}
    for result in results:
        score = _extract_score(result, logger)
        scores[result.task_name] = score
        if not np.isnan(score):
            logger.info(f"    {result.task_name:<20} {score:.4f}")
        else:
            logger.warning(f"    {result.task_name:<20} N/A")

    logger.info(f"  Average: {_avg(scores):.4f}")
    return scores


def _avg(scores: dict[str, float]) -> float:
    vals = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _short_name(model_path: str) -> str:
    return Path(model_path).name or model_path


# ─────────────────────────────────────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_txt(
    all_scores: dict[str, dict[str, float]],
    model_paths: list[str],
    out_path: Path,
    timestamp: str,
) -> None:
    col_w  = 14
    header = f"  {'Task':<24}" + "".join(f"  {_short_name(m):<{col_w}}" for m in model_paths)
    sep    = "  " + "-" * (len(header) - 2)
    lines  = [
        "=" * 76,
        "MTEB STS Evaluation",
        f"Date      : {timestamp}",
        "Benchmark : MTEB(eng, v1) — 10 STS tasks",
        "Metric    : cosine_spearman",
        "=" * 76, "", "Models", "-" * 76,
    ]
    for m in model_paths:
        lines.append(f"  {_short_name(m):<20} {m}")
    lines += ["", "Results", "-" * 76, header, sep]
    for task in MTEB_STS_TASKS:
        row = f"  {task:<24}"
        for m in model_paths:
            v = all_scores.get(m, {}).get(task, float("nan"))
            row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
        lines.append(row)
    lines.append(sep)
    avg_row = f"  {'Average (10 tasks)':<24}"
    for m in model_paths:
        avg_row += f"  {_avg(all_scores.get(m, {})):>{col_w}.4f}"
    lines += [avg_row, "", "=" * 76]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_md(
    all_scores: dict[str, dict[str, float]],
    model_paths: list[str],
    out_path: Path,
    timestamp: str,
) -> None:
    def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "N/A"
    cols_h   = " | ".join(_short_name(m) for m in model_paths)
    cols_sep = "|".join("------" for _ in model_paths)
    lines = [
        "# MTEB STS Evaluation",
        f"**Date**: {timestamp}  ",
        "**Benchmark**: MTEB(eng, v1) — 10 STS tasks  ",
        "**Metric**: `cosine_spearman`",
        "", "## Models", "",
    ]
    for m in model_paths:
        lines.append(f"- `{m}`")
    lines += [
        "", "## Results", "",
        "![scores](charts/scores.png)",
        "",
        f"| Task | {cols_h} |",
        f"|------|{cols_sep}|",
    ]
    for task in MTEB_STS_TASKS:
        row = f"| {task} |"
        for m in model_paths:
            row += " " + fmt(all_scores.get(m, {}).get(task, float("nan"))) + " |"
        lines.append(row)
    avg_row = "| **Average** |"
    for m in model_paths:
        avg_row += " **" + fmt(_avg(all_scores.get(m, {}))) + "** |"
    lines.append(avg_row)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_scores(
    all_scores: dict[str, dict[str, float]],
    model_paths: list[str],
    out_path: Path,
) -> None:
    tasks_display = MTEB_STS_TASKS + ["Average"]
    n_models = len(model_paths)
    x = np.arange(len(tasks_display))
    width = min(0.72 / n_models, 0.22)

    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#888888", "axes.grid": True,
        "grid.color": "#EBEBEB", "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(max(14, len(tasks_display) * 1.3), 6))

    for i, (m, color) in enumerate(zip(model_paths, COLORS)):
        task_vals = [all_scores.get(m, {}).get(t, float("nan")) for t in MTEB_STS_TASKS]
        vals = task_vals + [_avg(dict(zip(MTEB_STS_TASKS, task_vals)))]
        bars = ax.bar(x + (i - n_models / 2 + 0.5) * width, vals, width,
                      label=_short_name(m), color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.axvline(x=len(tasks_display) - 1 - 0.5, color="#AAAAAA", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_display, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("cosine_spearman")
    ax.set_title("MTEB(eng, v1) STS — cosine_spearman")
    ax.legend(fontsize=8, loc="lower right", ncol=max(1, n_models // 3))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models on MTEB(eng, v1) STS (cosine_spearman).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", dest="models", action="append", required=True, metavar="MODEL",
        help="模型路径或 HF model ID（可多次指定）",
    )
    args = parser.parse_args()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = RESULTS_DIR / timestamp
    charts_dir = out_dir / "charts"
    raw_dir    = out_dir / "raw"
    for d in [out_dir, charts_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "eval.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Models : {args.models}")
    logger.info(f"Output : {out_dir}")

    all_scores: dict[str, dict[str, float]] = {}
    for model_path in args.models:
        logger.info(f"\n{'─'*50}\nModel: {model_path}")
        try:
            all_scores[model_path] = run_mteb_eval(model_path, MTEB_STS_TASKS, raw_dir, logger)
        except Exception as exc:
            logger.error(f"  FAILED: {exc}", exc_info=True)
            all_scores[model_path] = {}

    # 打印汇总表
    col_w  = 14
    header = f"  {'Task':<24}" + "".join(f"  {_short_name(m):<{col_w}}" for m in args.models)
    sep    = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*70}\n  MTEB(eng, v1) STS — cosine_spearman\n{'='*70}")
    print(header); print(sep)
    for task in MTEB_STS_TASKS:
        row = f"  {task:<24}"
        for m in args.models:
            v = all_scores.get(m, {}).get(task, float("nan"))
            row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
        print(row)
    print(sep)
    avg_row = f"  {'Average (10 tasks)':<24}"
    for m in args.models:
        avg_row += f"  {_avg(all_scores.get(m, {})):>{col_w}.4f}"
    print(avg_row)
    print(f"{'='*70}\n")

    scores_path = raw_dir / "scores.json"
    scores_path.write_text(
        json.dumps({"timestamp": timestamp, "models": args.models,
                    "scores": all_scores,
                    "averages": {m: _avg(s) for m, s in all_scores.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Scores  → {scores_path}")

    try:
        write_summary_txt(all_scores, args.models, out_dir / "summary.txt", timestamp)
        logger.info(f"Summary → {out_dir / 'summary.txt'}")
    except Exception as exc:
        logger.error(f"summary.txt failed: {exc}")

    try:
        write_report_md(all_scores, args.models, out_dir / "report.md", timestamp)
        logger.info(f"Report  → {out_dir / 'report.md'}")
    except Exception as exc:
        logger.error(f"report.md failed: {exc}")

    try:
        plot_scores(all_scores, args.models, charts_dir / "scores.png")
        logger.info(f"Plot    → {charts_dir / 'scores.png'}")
    except Exception as exc:
        logger.error(f"Plot failed: {exc}")

    logger.info(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    main()
