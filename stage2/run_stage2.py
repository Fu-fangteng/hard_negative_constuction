#!/usr/bin/env python3
"""
Stage 2 主运行脚本
==================
从 nli_for_simcse 数据构造四元组 {anchor, pos, neg, hard_neg}。

用法示例：
  python stage2/run_stage2.py \\
      --input_path data/raw/nli_train.parquet \\
      --output_base data/stage2 \\
      --sample_size 1000 \\
      --methods all \\
      --recognizer regular

参数说明见 --help。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stage2.llm_engine import Qwen3Engine
from stage2.analyzer import (
    aggregate_final_dataset,
    build_dataset_methods_stat,
    generate_difference_report,
)
from stage2.builder import ALL_METHODS, PipelineRunner
from stage2.data_loader import load_data, save_preprocessed


def _parse_methods(s: str) -> list:
    if s.strip().lower() == "all":
        return list(ALL_METHODS)
    return [m.strip() for m in s.split(",") if m.strip()]


def _parse_recognizers(s: str) -> list:
    mapping = {"both": ["Regular", "LLM"], "regular": ["Regular"], "llm": ["LLM"]}
    return mapping.get(s.lower(), ["Regular"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: NLI Hard Negative Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_path",  required=True,
                        help="输入文件路径（.parquet 或 .jsonl）")
    parser.add_argument("--output_base", default="data/stage2",
                        help="输出根目录")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="采样数量（不指定则使用全部数据）")
    parser.add_argument("--methods",     default="all",
                        help="构造方法，逗号分隔或 'all'")
    parser.add_argument("--recognizer",  default="regular",
                        choices=["regular", "llm", "both"],
                        help="特征识别方式")
    parser.add_argument("--llm_model",      default="",
                        help="Qwen3 模型路径或 HF model ID（默认 Qwen/Qwen3-1.7B）")
    parser.add_argument("--llm_batch_size", type=int, default=16,
                        help="LLM 批推理大小，显存不足时调小（默认 16）")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    output_base  = Path(args.output_base)
    methods      = _parse_methods(args.methods)
    recognizers  = _parse_recognizers(args.recognizer)

    # ── 加载 LLM（如需要）───────────────────────────────────────────────
    llm_engine = None
    if "LLM" in recognizers:
        model_path = args.llm_model or "Qwen/Qwen3-1.7B"
        print(f"[LLM] Loading Qwen3: {model_path} ...")
        llm_engine = Qwen3Engine(model_name_or_path=model_path)
        llm_engine.load()
        print("[LLM] Ready.")

    # ── Step 1: 加载 & 预处理 ────────────────────────────────────────────
    print(f"\n[STEP 1] Loading data from {args.input_path}")
    records = load_data(args.input_path, sample_size=args.sample_size, seed=args.seed)
    print(f"  → {len(records)} samples loaded")

    preprocessed_path = output_base / "preprocessed" / "preprocessed_data.json"
    save_preprocessed(records, str(preprocessed_path))
    print(f"  → Saved to {preprocessed_path}")

    # ── Step 2: 各方法 × 识别器运行 ────────────────────────────────────
    all_results: dict = {}

    for method in methods:
        all_results[method] = {}
        for recognizer in recognizers:
            out_dir = output_base / "processed" / method / recognizer
            print(f"\n[RUN] {method} × {recognizer}")
            runner = PipelineRunner(
                records=records,
                method_name=method,
                recognizer_type=recognizer,
                output_dir=out_dir,
                llm_engine=llm_engine if recognizer == "LLM" else None,
                llm_batch_size=args.llm_batch_size,
            )
            result = runner.run()
            all_results[method][recognizer] = result
            s = result.stats
            print(f"  → {s['success_count']}/{s['total_samples']} success "
                  f"({s['success_ratio']*100:.1f}%)  time={s['processing_time_sec']}s")

        # difference.md（当两条路径都跑了时）
        if "Regular" in all_results[method] and "LLM" in all_results[method]:
            diff = generate_difference_report(
                method,
                all_results[method]["Regular"],
                all_results[method]["LLM"],
            )
            diff_path = output_base / "processed" / method / "difference.md"
            diff_path.write_text(diff, encoding="utf-8")
            print(f"  → difference.md written")

    # ── Step 3: 全局汇总 ────────────────────────────────────────────────
    print("\n[STEP 3] Aggregating global stats ...")

    record_ids = [r.id for r in records]
    record_pos = {r.id: r.pos for r in records}

    stat_rows = build_dataset_methods_stat(record_ids, record_pos, all_results)
    stat_path = output_base / "processed" / "dataset_methods_stat.json"
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    with stat_path.open("w", encoding="utf-8") as f:
        json.dump(stat_rows, f, ensure_ascii=False, indent=2)
    print(f"  → dataset_methods_stat.json: {len(stat_rows)} samples")

    final_rows = aggregate_final_dataset(all_results)
    final_path = output_base / "processed" / "final_dataset.jsonl"
    with final_path.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  → final_dataset.jsonl: {len(final_rows)} unique samples")

    print(f"\n✅ Stage 2 complete!")
    print(f"📁 Output: {output_base / 'processed'}")


if __name__ == "__main__":
    main()
