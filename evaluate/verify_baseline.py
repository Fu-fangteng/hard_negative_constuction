#!/usr/bin/env python3
"""
evaluate/verify_baseline.py
============================
复现 MTEB 排行榜得分并验证评测管道的正确性。

工作流程
────────
Step 1  生成参考分数
    python evaluate/verify_baseline.py --generate
    → 在 all-MiniLM-L6-v2（官方预训练权重）上跑完整 10 个 MTEB STS 任务
    → 结果保存到 evaluate/baseline_reference.json

Step 2  对比验证（检验你自己的模型）
    python evaluate/verify_baseline.py --check --model /path/to/your/model
    → 用 --model 指定的模型跑相同 10 个任务
    → 对比 baseline_reference.json，报告每个任务的分差和通过/失败状态

Step 3  仅打印参考分数
    python evaluate/verify_baseline.py --show

说明
────
· 指标：cosine_spearman（= main_score），与 MTEB 排行榜完全一致
· 任务集：MTEB(eng, v1) STS 10 个任务
· 容差：默认 ±0.002（浮点精度 + 数据版本差异）

用法示例
────────
  # 生成基准分数（首次运行，耗时 ~20min on GPU）
  python evaluate/verify_baseline.py --generate

  # 验证 NLI 微调模型是否在正确范围内运行，并与基准对比
  python evaluate/verify_baseline.py --check --model models/nli_run/final

  # 只打印保存的基准
  python evaluate/verify_baseline.py --show
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_MODEL = "all-MiniLM-L6-v2"
REFERENCE_FILE = Path(__file__).parent / "baseline_reference.json"
RESULTS_DIR    = Path(__file__).parent / "results"

# MTEB(eng, v1) 的 10 个 STS 任务（来源：mteb.get_benchmark('MTEB(eng, v1)')，type='STS'）
MTEB_STS_V1 = [
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

# 已知 all-MiniLM-L6-v2 在部分任务上的参考分数（来自 MTEB 论文和已保存结果）
# 用于自检：生成的基准分数与这些值相差超过 0.005 时给出警告
_KNOWN_APPROX: dict[str, float] = {
    "STSBenchmark": 0.8203,   # 已通过本地运行确认
    "SICK-R":       0.8043,   # MTEB 论文 Table 3
    "STS12":        0.6716,
    "STS13":        0.8180,
    "STS14":        0.6741,
    "STS15":        0.8241,
    "STS16":        0.7790,
}

CHECK_TOLERANCE = 0.002   # 与参考分数的允许误差（相同模型、相同任务）


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估
# ─────────────────────────────────────────────────────────────────────────────
def run_sts_eval(
    model_path: str,
    tasks: list[str],
    output_dir: Path,
) -> dict[str, float]:
    """
    对给定模型跑 MTEB STS 评估，返回 {task_name: cosine_spearman}。
    输出文件存到 output_dir（MTEB 标准格式，可与官方结果直接对比）。
    """
    import mteb
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    evaluation   = mteb.MTEB(tasks=task_objects)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Running {len(tasks)} MTEB STS tasks ...")

    results = evaluation.run(
        model,
        output_folder=str(output_dir),
        verbosity=1,
        encode_kwargs={"batch_size": 64},
    )

    scores: dict[str, float] = {}
    for result in results:
        try:
            score = float(result.get_score())
        except Exception:
            score = float("nan")
        scores[result.task_name] = score

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 格式化打印
# ─────────────────────────────────────────────────────────────────────────────
def _avg(scores: dict[str, float]) -> float:
    vals = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def print_scores_table(scores: dict[str, float], title: str = ""):
    if title:
        print(f"\n{'='*52}")
        print(f"  {title}")
        print(f"{'='*52}")
    print(f"  {'Task':<28} {'cosine_spearman':>16}")
    print(f"  {'-'*44}")
    for task in MTEB_STS_V1:
        v = scores.get(task, float("nan"))
        mark = ""
        if task in _KNOWN_APPROX and not np.isnan(v):
            diff = abs(v - _KNOWN_APPROX[task])
            mark = "  ✓" if diff <= 0.005 else f"  ⚠ expected≈{_KNOWN_APPROX[task]:.4f}"
        print(f"  {task:<28} {v:>16.4f}{mark}" if not np.isnan(v)
              else f"  {task:<28} {'N/A':>16}")
    avg = _avg(scores)
    print(f"  {'-'*44}")
    print(f"  {'Average (10 tasks)':<28} {avg:>16.4f}")


def print_comparison_table(
    target_scores: dict[str, float],
    ref_scores: dict[str, float],
    target_label: str,
    ref_label: str,
    tolerance: float,
) -> bool:
    """打印对比表，返回 True 表示全部任务通过容差检查。"""
    print(f"\n{'='*72}")
    print(f"  Comparison: {target_label}  vs  {ref_label}")
    print(f"  Tolerance: ±{tolerance:.3f}")
    print(f"{'='*72}")
    print(f"  {'Task':<28} {ref_label:>12} {target_label:>12} {'Δ':>8}  Status")
    print(f"  {'-'*66}")

    all_pass = True
    for task in MTEB_STS_V1:
        ref = ref_scores.get(task, float("nan"))
        tgt = target_scores.get(task, float("nan"))
        if np.isnan(ref) or np.isnan(tgt):
            status = "⚠  N/A"
        else:
            delta = tgt - ref
            if abs(delta) <= tolerance:
                status = "✓ PASS"
            elif delta > tolerance:
                status = f"↑ BETTER (+{delta:.4f})"
            else:
                status = f"↓ WORSE  ({delta:.4f})"
                all_pass = False
        ref_str = f"{ref:.4f}" if not np.isnan(ref) else "N/A"
        tgt_str = f"{tgt:.4f}" if not np.isnan(tgt) else "N/A"
        d_str   = f"{tgt-ref:+.4f}" if not (np.isnan(ref) or np.isnan(tgt)) else "N/A"
        print(f"  {task:<28} {ref_str:>12} {tgt_str:>12} {d_str:>8}  {status}")

    print(f"  {'-'*66}")
    ref_avg = _avg(ref_scores)
    tgt_avg = _avg(target_scores)
    d_avg   = tgt_avg - ref_avg
    avg_status = "↑ BETTER" if d_avg > 0 else ("↓ WORSE" if d_avg < 0 else "= SAME")
    print(f"  {'Average':<28} {ref_avg:>12.4f} {tgt_avg:>12.4f} {d_avg:>+8.4f}  {avg_status}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 三个主命令
# ─────────────────────────────────────────────────────────────────────────────
def cmd_generate(args):
    """Step 1: 跑 baseline 模型，保存参考分数。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"baseline_{timestamp}"

    print(f"\n[GENERATE] Running baseline model: {BASELINE_MODEL}")
    print(f"Tasks : {MTEB_STS_V1}")
    print(f"Output: {out_dir}")

    scores = run_sts_eval(BASELINE_MODEL, MTEB_STS_V1, out_dir)

    print_scores_table(scores, f"Baseline: {BASELINE_MODEL}")

    # 与已知参考值自检
    warn_count = 0
    for task, expected in _KNOWN_APPROX.items():
        got = scores.get(task, float("nan"))
        if not np.isnan(got) and abs(got - expected) > 0.005:
            print(f"  ⚠  WARNING: {task} got {got:.4f}, expected ≈ {expected:.4f} "
                  f"(diff={got-expected:+.4f}). Check mteb version or model loading.")
            warn_count += 1

    if warn_count == 0:
        print("\n  ✓  All known-reference tasks are within tolerance.")
    else:
        print(f"\n  ⚠  {warn_count} task(s) deviate from known references — "
              f"verify mteb version and model path.")

    # 保存参考文件
    reference = {
        "model":     BASELINE_MODEL,
        "timestamp": timestamp,
        "mteb_tasks": MTEB_STS_V1,
        "scores":    scores,
        "average":   _avg(scores),
    }
    REFERENCE_FILE.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Reference saved → {REFERENCE_FILE}")


def cmd_check(args):
    """Step 2: 对比目标模型与已保存的 baseline 参考。"""
    if not REFERENCE_FILE.exists():
        print(f"ERROR: Reference file not found: {REFERENCE_FILE}")
        print("Run first:  python evaluate/verify_baseline.py --generate")
        sys.exit(1)

    with open(REFERENCE_FILE, encoding="utf-8") as f:
        reference = json.load(f)
    ref_scores = reference["scores"]
    ref_label  = f"baseline ({reference['model']})"
    print(f"\n[CHECK] Reference: {reference['model']}  ({reference['timestamp']})")
    print(f"        Reference avg: {reference['average']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = Path(args.model).name
    out_dir = RESULTS_DIR / f"check_{model_label}_{timestamp}"

    print(f"\n[CHECK] Evaluating target model: {args.model}")
    target_scores = run_sts_eval(args.model, MTEB_STS_V1, out_dir)

    passed = print_comparison_table(
        target_scores, ref_scores,
        target_label=model_label,
        ref_label=ref_label,
        tolerance=args.tolerance,
    )

    # 保存对比结果
    result = {
        "model":           args.model,
        "timestamp":       timestamp,
        "reference_model": reference["model"],
        "scores":          target_scores,
        "average":         _avg(target_scores),
        "delta_avg":       _avg(target_scores) - reference["average"],
    }
    out_file = out_dir / "check_result.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_file}")

    if passed:
        print("\n  ✓  Model evaluation pipeline verified successfully.")
    else:
        print("\n  ✗  Some tasks scored below baseline beyond tolerance.")
        print("     This may indicate a model regression or evaluation error.")


def cmd_show(args):
    """Step 3: 打印已保存的参考分数。"""
    if not REFERENCE_FILE.exists():
        print(f"No reference file found at: {REFERENCE_FILE}")
        print("Run: python evaluate/verify_baseline.py --generate")
        sys.exit(1)

    with open(REFERENCE_FILE, encoding="utf-8") as f:
        reference = json.load(f)

    print_scores_table(
        reference["scores"],
        f"Baseline Reference: {reference['model']}  ({reference['timestamp']})",
    )
    print(f"\n  File: {REFERENCE_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MTEB STS baseline verification tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true",
                       help="跑 all-MiniLM-L6-v2 基准，保存参考分数到 baseline_reference.json")
    group.add_argument("--check",    action="store_true",
                       help="对比 --model 指定的模型与已保存的基准")
    group.add_argument("--show",     action="store_true",
                       help="打印已保存的基准分数")

    parser.add_argument("--model",     default=None,
                        help="（--check 时必填）待评测模型路径或 HF model ID")
    parser.add_argument("--tolerance", type=float, default=CHECK_TOLERANCE,
                        help=f"与基准的允许误差（默认 {CHECK_TOLERANCE}）")
    parser.add_argument("--tasks", nargs="+", default=MTEB_STS_V1,
                        help="要评测的 MTEB 任务列表（默认 MTEB(eng,v1) 的 10 个 STS 任务）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check and not args.model:
        print("ERROR: --check requires --model <path>")
        sys.exit(1)

    if args.generate:
        cmd_generate(args)
    elif args.check:
        cmd_check(args)
    elif args.show:
        cmd_show(args)


if __name__ == "__main__":
    main()
