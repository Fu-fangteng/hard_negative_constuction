#!/usr/bin/env python3
"""
evaluate/verify_baseline.py
============================
在 MTEB(eng, v1) 的 10 个 STS 任务上评测任意模型。
指标：cosine_spearman，与 MTEB 排行榜完全一致。

用法
────
  # 评测单个模型
  python evaluate/verify_baseline.py --model Qwen/Qwen3-Embedding-4B

  # 同时评测多个模型并对比
  python evaluate/verify_baseline.py \
      --model Qwen/Qwen3-Embedding-4B \
      --model all-MiniLM-L6-v2 \
      --model models/my_finetuned/final

输出（evaluate/results/{timestamp}/）
  scores.json   — 每个模型的 per-task 分数
  report.md     — Markdown 表格
  scores.png    — 柱状对比图
  eval.log      — 完整运行日志
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

# 已知 all-MiniLM-L6-v2 的参考分数，用于自动 sanity check
_KNOWN_APPROX: dict[str, dict[str, float]] = {
    "all-MiniLM-L6-v2": {
        "STSBenchmark": 0.8203,
        "SICK-R":       0.8043,
        "STS12":        0.6716,
        "STS13":        0.8180,
        "STS14":        0.6741,
        "STS15":        0.8241,
        "STS16":        0.7790,
    },
}

COLORS = ["#378ADD", "#1D9E75", "#D94F3D", "#BA7517", "#9B59B6",
          "#E67E22", "#27AE60", "#2980B9", "#8E44AD", "#C0392B"]


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估
# ─────────────────────────────────────────────────────────────────────────────
def _extract_score(result, logger: logging.Logger) -> float:
    """从 MTEB TaskResult 中提取 cosine_spearman，主路径失败时走 fallback。"""
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


def run_sts_eval(
    model_path: str,
    tasks: list[str],
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    逐任务运行 MTEB STS 评估，返回 {task_name: cosine_spearman}。
    遇到 OOM 时自动缩小 batch_size 重试，单任务失败不影响其他任务。
    """
    import mteb
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"  Loading: {model_path}")
    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "auto"},   # bf16/fp16，节省显存
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Running {len(tasks)} MTEB STS tasks (one by one) ...")

    scores: dict[str, float] = {}
    for task_name in tasks:
        task_score = float("nan")
        for batch_size in [32, 16, 8, 4]:
            try:
                task_objs  = mteb.get_tasks(tasks=[task_name], languages=["eng"])
                evaluation = mteb.MTEB(tasks=task_objs)
                results    = evaluation.run(
                    model,
                    output_folder=str(output_dir),
                    verbosity=0,
                    encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
                )
                task_score = _extract_score(results[0], logger)
                tag = f"  [batch={batch_size}]" if batch_size < 32 else ""
                if not np.isnan(task_score):
                    logger.info(f"    {task_name:<20} {task_score:.4f}{tag}")
                else:
                    logger.warning(f"    {task_name:<20} N/A{tag}")
                break   # 成功，继续下一个任务

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"    OOM at batch_size={batch_size} for {task_name}, retrying ...")
                torch.cuda.empty_cache()
                if batch_size == 4:
                    logger.error(f"    {task_name}: OOM at minimum batch_size=4, skipping")
            except Exception as exc:
                logger.error(f"    {task_name} failed: {exc}", exc_info=True)
                break

        scores[task_name] = task_score

    avg = _avg(scores)
    logger.info(f"  Average ({len(scores)} tasks): {avg:.4f}")
    return scores


def _avg(scores: dict[str, float]) -> float:
    vals = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _short_name(model_path: str) -> str:
    """取模型路径最后一段作为短标签。"""
    return Path(model_path).name or model_path


# ─────────────────────────────────────────────────────────────────────────────
# 打印 & 报告
# ─────────────────────────────────────────────────────────────────────────────
def print_table(
    all_scores: dict[str, dict[str, float]],
    model_names: list[str],
) -> None:
    col_w = 12
    header = f"  {'Task':<22}" + "".join(f"  {_short_name(m):>{col_w}}" for m in model_names)
    sep    = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*70}")
    print("  MTEB(eng, v1) STS — cosine_spearman")
    print(f"{'='*70}")
    print(header)
    print(sep)
    for task in MTEB_STS_TASKS:
        row = f"  {task:<22}"
        for m in model_names:
            v = all_scores.get(m, {}).get(task, float("nan"))
            row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'N/A':>{col_w}}"
        print(row)
    print(sep)
    avg_row = f"  {'Average':<22}"
    for m in model_names:
        avg = _avg(all_scores.get(m, {}))
        avg_row += f"  {avg:>{col_w}.4f}" if not np.isnan(avg) else f"  {'N/A':>{col_w}}"
    print(avg_row)
    print(f"{'='*70}\n")


def write_report(
    all_scores: dict[str, dict[str, float]],
    model_names: list[str],
    out_path: Path,
    timestamp: str,
) -> None:
    def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "N/A"

    cols_header = " | ".join(_short_name(m) for m in model_names)
    cols_sep    = "|".join("------" for _ in model_names)
    lines = [
        "# MTEB STS Evaluation",
        f"**Date**: {timestamp}  ",
        "**Benchmark**: MTEB(eng, v1) — 10 STS tasks  ",
        "**Metric**: `cosine_spearman`",
        "",
        "## Models",
        "",
    ]
    for m in model_names:
        lines.append(f"- `{m}`")
    lines += [
        "",
        "## Results",
        "",
        f"| Task | {cols_header} |",
        f"|------|{cols_sep}|",
    ]
    for task in MTEB_STS_TASKS:
        row = f"| {task} |"
        for m in model_names:
            row += " " + fmt(all_scores.get(m, {}).get(task, float("nan"))) + " |"
        lines.append(row)
    avg_row = "| **Average** |"
    for m in model_names:
        avg_row += " **" + fmt(_avg(all_scores.get(m, {}))) + "** |"
    lines.append(avg_row)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_scores(
    all_scores: dict[str, dict[str, float]],
    model_names: list[str],
    out_path: Path,
) -> None:
    tasks_display = MTEB_STS_TASKS + ["Average"]
    n_models = len(model_names)
    n_tasks  = len(tasks_display)
    x = np.arange(n_tasks)
    width = min(0.7 / n_models, 0.25)

    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#888888", "axes.grid": True,
        "grid.color": "#EBEBEB", "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans", "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(max(14, n_tasks * 1.2), 5.5))

    for i, (m, color) in enumerate(zip(model_names, COLORS)):
        vals = [all_scores.get(m, {}).get(t, float("nan")) for t in MTEB_STS_TASKS]
        vals_display = vals + [_avg(dict(zip(MTEB_STS_TASKS, vals)))]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals_display, width,
                      label=_short_name(m), color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals_display):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)

    # 在 Average 列前画分隔线
    ax.axvline(x=n_tasks - 1 - 0.5, color="#AAAAAA", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_display, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("cosine_spearman")
    ax.set_title("MTEB(eng, v1) STS Benchmark — cosine_spearman", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
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
        help="模型路径或 HF model ID（可多次指定以对比多个模型）",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_path = out_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Models  : {args.models}")
    logger.info(f"Tasks   : {MTEB_STS_TASKS}")
    logger.info(f"Output  : {out_dir}")

    all_scores: dict[str, dict[str, float]] = {}

    for model_path in args.models:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Model: {model_path}")
        raw_dir = out_dir / "raw" / _short_name(model_path)
        try:
            scores = run_sts_eval(model_path, MTEB_STS_TASKS, raw_dir, logger)
            all_scores[model_path] = scores

            # sanity check for known models
            base_name = _short_name(model_path)
            ref = _KNOWN_APPROX.get(base_name) or _KNOWN_APPROX.get(model_path)
            if ref:
                warn = [(t, scores[t], ref[t]) for t in ref
                        if t in scores and not np.isnan(scores[t])
                        and abs(scores[t] - ref[t]) > 0.005]
                if warn:
                    for t, got, exp in warn:
                        logger.warning(f"  ⚠ {t}: got {got:.4f}, expected ≈ {exp:.4f}")
                else:
                    logger.info("  ✓ Known-reference tasks all within ±0.005")

        except Exception as exc:
            logger.error(f"  FAILED: {exc}", exc_info=True)
            all_scores[model_path] = {}

    # 打印汇总表
    print_table(all_scores, args.models)

    # 保存 JSON
    scores_path = out_dir / "scores.json"
    scores_path.write_text(
        json.dumps({"timestamp": timestamp, "models": args.models,
                    "scores": all_scores,
                    "averages": {m: _avg(s) for m, s in all_scores.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Scores  → {scores_path}")

    # 报告
    report_path = out_dir / "report.md"
    write_report(all_scores, args.models, report_path, timestamp)
    logger.info(f"Report  → {report_path}")

    # 图表
    plot_path = out_dir / "scores.png"
    try:
        plot_scores(all_scores, args.models, plot_path)
        logger.info(f"Plot    → {plot_path}")
    except Exception as exc:
        logger.error(f"Plot failed: {exc}")

    logger.info(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    main()
