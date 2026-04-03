from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stage1.data_utils import export_records, load_data
from stage1.evaluator import evaluate_dataset
from stage1.formatter import build_methods_stat, export_formatter_outputs, format_dataset
from stage1.main_generator import generate_dataset
from stage1.sampler import select_top_k_by_score
from stage1.llm_engine import LocalLLMEngine, GenerationConfig


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
    parser.add_argument("--out_dir", type=str, default="data/stage1", help="Output directory")
    parser.add_argument("--k", type=int, default=100, help="Top-k by score to build dataset")
    parser.add_argument("--methods", type=str, default="auto",
                        help="Comma separated method names or 'auto'")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run similarity evaluation (requires angle_emb)")
    parser.add_argument("--similarity_model", type=str, default="WhereIsAI/UAE-Large-V1")
    parser.add_argument("--llm_model", type=str, default="")
    parser.add_argument("--llm_max_new_tokens", type=int, default=256)
    parser.add_argument("--llm_temperature", type=float, default=0.2)
    parser.add_argument("--gt_original_range", type=float, nargs=2, default=[0, 5])
    parser.add_argument("--gt_target_range", type=float, nargs=2, default=[0, 1])
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_methods_arg(args.methods)

    print("\n[STAGE 1] Loading and sampling data...")
    records = load_data(input_path)
    top_k = select_top_k_by_score(records, k=args.k)
    print(f"  → Selected top {len(top_k)} records by score")
    export_records(top_k, json_path=data_dir / "topk_positives.json",
                   csv_path=data_dir / "topk_positives.csv")

    print("\n[STAGE 2] Feature extraction and formatting...")
    llm_engine = None
    if args.llm_model:
        try:
            cfg = GenerationConfig(max_new_tokens=args.llm_max_new_tokens,
                                   temperature=args.llm_temperature)
            llm_engine = LocalLLMEngine(model_name_or_path=args.llm_model, default_config=cfg)
            llm_engine.load()
            print(f"  → LLM engine loaded: {args.llm_model}")
        except Exception as exc:
            print(f"  [WARN] LLM disabled: {exc}")

    formatted_data = format_dataset(top_k, llm_engine=llm_engine)
    methods_stat = build_methods_stat(formatted_data)
    export_formatter_outputs(
        formatted_data=formatted_data, methods_stat=methods_stat,
        formatted_data_path=data_dir / "formatted_data.json",
        methods_stat_path=data_dir / "methods_stat.json",
    )

    print("\n[STAGE 3] Constructing hard negatives...")
    dataset_rows = generate_dataset(formatted_data, methods=methods, llm_engine=llm_engine)
    print(f"  → Generated {len(dataset_rows)} hard negative samples")
    _export_jsonl(dataset_rows, data_dir / "final_dataset.jsonl")
    _export_final_csv(dataset_rows, data_dir / "final_dataset.csv")

    if args.evaluate:
        print("\n[STAGE 4] Evaluating similarities...")
        metrics = evaluate_dataset(
            dataset_rows, out_dir=out_dir,
            similarity_model=args.similarity_model,
            gt_original_range=tuple(args.gt_original_range),
            gt_target_range=tuple(args.gt_target_range),
        )
        print(f"  S1 mean={metrics['S1_stats']['mean']:.4f}  "
              f"S2 mean={metrics['S2_stats']['mean']:.4f}  "
              f"Gap mean={metrics['Gap_stats']['mean']:.4f}  "
              f"Validity={metrics['validity_ratio_S2_lt_S1']:.2%}")

    print(f"\n✅ Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
