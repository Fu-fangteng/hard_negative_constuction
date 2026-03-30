from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import export_records, load_data
from src.evaluator import evaluate_dataset
from src.formatter import build_methods_stat, export_formatter_outputs, format_dataset
from src.main_generator import generate_dataset
from src.sampler import select_top_k_by_score
from src.llm_engine import LocalLLMEngine, GenerationConfig


def _export_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_final_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text1", "text2", "text3", "score", "methods_used"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            r = dict(row)
            r["methods_used"] = ",".join(row.get("methods_used") or [])
            writer.writerow(r)


def parse_methods_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() == "auto":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="STS Hard Negative Construction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input data path (.jsonl/.json/.csv)")
    parser.add_argument("--out_dir", type=str, default="outputs/run_1", help="Output directory")
    parser.add_argument("--k", type=int, default=100, help="Top-k by score to build dataset")
    parser.add_argument(
        "--methods",
        type=str,
        default="auto",
        help="Comma separated method names or 'auto'. Example: direct_negation_attack,entity_pronoun_substitution",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run similarity evaluation (requires angle_emb)")
    parser.add_argument(
        "--similarity_model",
        type=str,
        default="WhereIsAI/UAE-Large-V1",
        help="angle_emb model name for similarity computation",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="",
        help="Optional: enable LLM-based feature extraction in formatter by providing a local model path/name.",
    )
    parser.add_argument("--llm_max_new_tokens", type=int, default=256, help="LLM max_new_tokens for feature extraction")
    parser.add_argument("--llm_temperature", type=float, default=0.2, help="LLM temperature for feature extraction")
    parser.add_argument(
        "--gt_original_range",
        type=float,
        nargs=2,
        default=[0, 5],
        help="Original GT score range (min max). Default: 0 5"
    )
    parser.add_argument(
        "--gt_target_range",
        type=float,
        nargs=2,
        default=[0, 1],
        help="Target GT score range (min max) after mapping. Default: 0 1"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create organized subdirectories for pipeline stages
    data_dir = out_dir / "data"
    metrics_dir = out_dir / "metrics"
    reports_dir = out_dir / "reports"
    visuals_dir = out_dir / "visualizations"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INIT] Output directory structure created at: {out_dir}")

    methods = parse_methods_arg(args.methods)

    # 1) Load raw -> top-k positives
    print("\n[STAGE 1] Loading and sampling data...")
    records = load_data(input_path)
    top_k = select_top_k_by_score(records, k=args.k)
    print(f"  → Selected top {len(top_k)} records by score")

    # Export top-k positives to data directory
    export_records(
        top_k,
        json_path=data_dir / "topk_positives.json",
        csv_path=data_dir / "topk_positives.csv",
    )
    print(f"  → Saved top-k positives to {data_dir}")

    # 2) Feature extraction/formatting
    print("\n[STAGE 2] Feature extraction and formatting...")
    llm_engine = None
    if args.llm_model:
        try:
            cfg = GenerationConfig(max_new_tokens=args.llm_max_new_tokens, temperature=args.llm_temperature)
            llm_engine = LocalLLMEngine(model_name_or_path=args.llm_model, default_config=cfg)
            llm_engine.load()
            print(f"  → LLM engine loaded: {args.llm_model}")
        except Exception as exc:
            print(f"  [WARN] LLM feature extraction disabled: {exc}")
            llm_engine = None

    formatted_data = format_dataset(top_k, llm_engine=llm_engine)
    methods_stat = build_methods_stat(formatted_data)
    
    export_formatter_outputs(
        formatted_data=formatted_data,
        methods_stat=methods_stat,
        formatted_data_path=data_dir / "formatted_data.json",
        methods_stat_path=data_dir / "methods_stat.json",
    )
    print(f"  → Saved formatted data to {data_dir}")

    # 3) Construct hard negatives (text3)
    print("\n[STAGE 3] Constructing hard negatives...")
    dataset_rows = generate_dataset(formatted_data, methods=methods, llm_engine=llm_engine)
    print(f"  → Generated {len(dataset_rows)} hard negative samples")

    _export_jsonl(dataset_rows, data_dir / "final_dataset.jsonl")
    _export_final_csv(dataset_rows, data_dir / "final_dataset.csv")
    print(f"  → Saved final dataset to {data_dir}")

    # 4) Evaluate
    if args.evaluate:
        print("\n[STAGE 4] Evaluating similarities...")
        metrics = evaluate_dataset(
            dataset_rows,
            out_dir=out_dir,
            similarity_model=args.similarity_model,
            gt_original_range=tuple(args.gt_original_range),
            gt_target_range=tuple(args.gt_target_range),
        )
        
        # Also save to reports directory for summary
        with (reports_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "n_samples": metrics["n"],
                "similarity_backend": metrics["similarity_backend"],
                "S1_stats": metrics["S1_stats"],
                "S2_stats": metrics["S2_stats"],
                "Gap_stats": metrics["Gap_stats"],
                "validity_ratio": metrics["validity_ratio_S2_lt_S1"],
                "best_methods": metrics.get("best_methods_by_mean_gap", [])[:3]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"  → Evaluation complete")
        print(f"\n[RESULTS]")
        print(f"  📊 Samples: {metrics['n']}")
        print(f"  📈 S1 (original) - Mean: {metrics['S1_stats']['mean']:.4f}, Median: {metrics['S1_stats']['median']:.4f}")
        print(f"  📉 S2 (generated) - Mean: {metrics['S2_stats']['mean']:.4f}, Median: {metrics['S2_stats']['median']:.4f}")
        print(f"  📌 Gap (S1-S2) - Mean: {metrics['Gap_stats']['mean']:.4f}, Median: {metrics['Gap_stats']['median']:.4f}")
        print(f"  ✓ Validity (S2 < S1): {metrics['validity_ratio_S2_lt_S1']:.2%}")
    else:
        print("\n[STAGE 4] Skipping evaluation (use --evaluate flag to run)")

    print(f"\n✅ Pipeline completed successfully!")
    print(f"📁 Output directory: {out_dir}")
    print(f"   ├── data/              (sampled data, formatted data, final dataset)")
    print(f"   ├── metrics/           (evaluation metrics & detailed results)")
    print(f"   ├── reports/           (markdown & text reports)")
    print(f"   └── visualizations/    (PNG charts)")


if __name__ == "__main__":
    main()
