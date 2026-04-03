from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .formatter import METHOD_NAMES
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    return n / d if d != 0 else default


def _to_numpy(x: Any):
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    # Last resort
    return np.asarray(x)


def _cosine_similarity_batch(a, b):
    """
    a: (n, d) embeddings
    b: (n, d) embeddings
    return: (n,)
    """
    import numpy as np

    a = _to_numpy(a)
    b = _to_numpy(b)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a = a / np.clip(a_norm, 1e-12, None)
    b = b / np.clip(b_norm, 1e-12, None)
    return (a * b).sum(axis=1)


def _bow_tokens(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", (text or "").lower())


def _bow_counter(tokens: Sequence[str]):
    from collections import Counter

    return Counter(tokens)


def _cosine_counter(a, b) -> float:
    # cosine similarity with term-frequency vectors.
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += float(va) * float(vb)
    na = sum(float(v) * float(v) for v in a.values()) ** 0.5
    nb = sum(float(v) * float(v) for v in b.values()) ** 0.5
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _similarity_bow(text_a: Sequence[str], text_b: Sequence[str]) -> List[float]:
    if len(text_a) != len(text_b):
        raise ValueError("text_a and text_b must have the same length")
    sims: List[float] = []
    vec_a = [_bow_counter(_bow_tokens(t)) for t in text_a]
    vec_b = [_bow_counter(_bow_tokens(t)) for t in text_b]
    for va, vb in zip(vec_a, vec_b):
        sims.append(_cosine_counter(va, vb))
    return sims


def _compute_basic_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if len(vals) == 0:
        return {"mean": 0.0, "var": 0.0, "median": 0.0}
    mean = statistics.fmean(vals)
    # population variance to match numpy default (ddof=0)
    var = statistics.pvariance(vals) if len(vals) >= 2 else 0.0
    median = statistics.median(vals)
    return {"mean": float(mean), "var": float(var), "median": float(median)}


def _map_gt_score(gt_score: float, original_range: Tuple[float, float] = (0, 5), target_range: Tuple[float, float] = (0, 1)) -> float:
    """
    将GT分数从原始范围映射到目标范围
    
    Args:
        gt_score: 原始GT分数
        original_range: 原始分数范围 (min, max)
        target_range: 目标分数范围 (min, max)
    
    Returns:
        映射后的分数
    """
    orig_min, orig_max = original_range
    target_min, target_max = target_range
    
    # 处理边界情况
    if orig_max == orig_min:
        return target_min
    
    # 线性映射
    normalized = (gt_score - orig_min) / (orig_max - orig_min)
    mapped = target_min + normalized * (target_max - target_min)
    
    # 裁剪到目标范围
    return max(target_min, min(target_max, mapped))


def _generate_text_report(
    metrics: Dict[str, Any],
    rows_data: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str = "eval"
) -> Dict[str, str]:
    """生成美观的文本报告（支持Markdown和纯文本格式）"""
    
    # 生成Markdown报告
    md_path = out_dir / f"{prefix}_report.md"
    txt_path = out_dir / f"{prefix}_report.txt"
    
    # 准备报告内容
    report_lines = []
    
    # 标题
    report_lines.append("# Evaluation Report\n")
    report_lines.append(f"**Generated:** {Path(__file__).stem}\n")
    report_lines.append(f"**Samples:** {metrics['n']:,}\n")
    report_lines.append(f"**Similarity Backend:** {metrics['similarity_backend']}\n")
    
    # 添加GT映射信息
    if metrics.get('gt_mapping'):
        gt_map = metrics['gt_mapping']
        report_lines.append(f"**GT Score Mapping:** {gt_map['original_range'][0]}–{gt_map['original_range'][1]} → {gt_map['target_range'][0]}–{gt_map['target_range'][1]}\n")
    
    report_lines.append("\n---\n")
    
    # 1. 核心指标表格
    report_lines.append("## 📊 Key Metrics\n")
    report_lines.append("| Metric | S1 (Text1-Text2) | S2 (Text1-Text3) | Gap (S1 - S2) |")
    report_lines.append("|--------|------------------|------------------|---------------|")
    
    stats_keys = ["mean", "median", "var"]
    stat_names = {"mean": "Mean", "median": "Median", "var": "Variance"}
    
    for stat_key in stats_keys:
        s1_val = metrics["S1_stats"][stat_key]
        s2_val = metrics["S2_stats"][stat_key]
        gap_val = metrics["Gap_stats"][stat_key]
        report_lines.append(f"| **{stat_names[stat_key]}** | {s1_val:.4f} | {s2_val:.4f} | {gap_val:.4f} |")
    
    report_lines.append(f"| **Validity Ratio** (S2 < S1) | - | - | {metrics['validity_ratio_S2_lt_S1']:.2%} |")
    report_lines.append("\n")
    
    # 2. GT偏移统计
    report_lines.append("## 🎯 Ground Truth Offset\n")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Mean Offset | {metrics['GT_offset_stats']['mean']:.4f} |")
    report_lines.append(f"| Median Offset | {metrics['GT_offset_stats']['median']:.4f} |")
    report_lines.append(f"| Variance | {metrics['GT_offset_stats']['var']:.4f} |")
    
    # 添加相关性指标
    if metrics.get('gt_correlation'):
        report_lines.append(f"| Correlation (S2 vs GT) | {metrics['gt_correlation']['pearson']:.4f} |")
        report_lines.append(f"| Spearman Correlation | {metrics['gt_correlation']['spearman']:.4f} |")
    
    report_lines.append("\n")
    
    # 3. 方法贡献度表格
    if metrics.get("method_contribution"):
        report_lines.append("## 🔬 Method Contribution Analysis\n")
        report_lines.append("| Method | Samples | Mean Gap | Median Gap | Variance |")
        report_lines.append("|--------|---------|----------|------------|----------|")
        
        # 按平均gap排序
        sorted_methods = sorted(
            metrics["method_contribution"].items(),
            key=lambda x: x[1]["gap_stats"]["mean"],
            reverse=True
        )
        
        for method_name, method_stats in sorted_methods:
            if method_stats["n"] > 0:
                stats = method_stats["gap_stats"]
                report_lines.append(
                    f"| {method_name} | {method_stats['n']} | "
                    f"{stats['mean']:.4f} | {stats['median']:.4f} | "
                    f"{stats['var']:.4f} |"
                )
        report_lines.append("\n")
    
    # 4. Top方法
    if metrics.get("best_methods_by_mean_gap"):
        report_lines.append("## 🏆 Top Methods by Mean Gap\n")
        report_lines.append("| Rank | Method | Mean Gap | Samples |")
        report_lines.append("|------|--------|----------|---------|")
        
        for i, method_info in enumerate(metrics["best_methods_by_mean_gap"], 1):
            report_lines.append(
                f"| {i} | {method_info['method']} | "
                f"{method_info['mean_gap']:.4f} | {method_info['n']} |"
            )
        report_lines.append("\n")
    
    # 5. 统计摘要
    report_lines.append("## 📈 Distribution Summary\n")
    report_lines.append("### S1 Distribution (Text1-Text2)\n")
    s1_values = [r.get('s1', 0) for r in rows_data]
    report_lines.append(f"- **Range:** [{min(s1_values):.4f}, {max(s1_values):.4f}]")
    report_lines.append(f"- **Mean ± Std:** {metrics['S1_stats']['mean']:.4f} ± {metrics['S1_stats']['var']**0.5:.4f}\n")
    
    report_lines.append("### S2 Distribution (Text1-Text3)\n")
    s2_values = [r.get('s2', 0) for r in rows_data]
    report_lines.append(f"- **Range:** [{min(s2_values):.4f}, {max(s2_values):.4f}]")
    report_lines.append(f"- **Mean ± Std:** {metrics['S2_stats']['mean']:.4f} ± {metrics['S2_stats']['var']**0.5:.4f}\n")
    
    report_lines.append("### Gap Distribution\n")
    gap_values = [r.get('gap', 0) for r in rows_data]
    report_lines.append(f"- **Range:** [{min(gap_values):.4f}, {max(gap_values):.4f}]")
    report_lines.append(f"- **Mean ± Std:** {metrics['Gap_stats']['mean']:.4f} ± {metrics['Gap_stats']['var']**0.5:.4f}")
    report_lines.append(f"- **Positive Gap Ratio:** {sum(1 for r in rows_data if r.get('gap', 0) > 0) / max(1, len(rows_data)):.2%}")
    report_lines.append("\n")
    
    # 6. GT分布统计
    if rows_data and 'gt_score_mapped' in rows_data[0]:
        report_lines.append("### GT Score Distribution (Mapped to 0-1)\n")
        gt_mapped = [r.get('gt_score_mapped', 0) for r in rows_data]
        report_lines.append(f"- **Range:** [{min(gt_mapped):.4f}, {max(gt_mapped):.4f}]")
        report_lines.append(f"- **Mean ± Std:** {statistics.mean(gt_mapped):.4f} ± {statistics.stdev(gt_mapped) if len(gt_mapped) > 1 else 0:.4f}")
        report_lines.append("\n")
    
    # 7. 示例数据（前10行）
    report_lines.append("## 📝 Sample Data (First 10 rows)\n")
    report_lines.append("| ID | Text1 (truncated) | Text2 (truncated) | Text3 (truncated) | S1 | S2 | Gap | GT (mapped) |")
    report_lines.append("|----|-------------------|-------------------|-------------------|----|----|-----|-------------|")
    
    for i, row in enumerate(rows_data[:10]):
        t1 = row['text1'][:50] + "..." if len(row['text1']) > 50 else row['text1']
        t2 = row['text2'][:50] + "..." if len(row['text2']) > 50 else row['text2']
        t3 = row['text3'][:50] + "..." if len(row['text3']) > 50 else row['text3']
        gt_display = f"{row.get('gt_score_mapped', row.get('gt_score', 0)):.4f}"
        report_lines.append(
            f"| {row['id']} | {t1} | {t2} | {t3} | "
            f"{row['s1']:.4f} | {row['s2']:.4f} | {row['gap']:.4f} | {gt_display} |"
        )
    report_lines.append("\n")
    
    # 写入Markdown文件
    md_content = "\n".join(report_lines)
    md_path.write_text(md_content, encoding="utf-8")
    
    # 同时生成纯文本版本（去除Markdown格式）
    txt_lines = []
    for line in report_lines:
        # 移除Markdown标题标记
        line = re.sub(r'^#+\s+', '', line)
        # 移除表格分隔线
        if line.strip().startswith('|---'):
            continue
        # 移除Markdown链接格式
        line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
        txt_lines.append(line)
    
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    
    return {
        "markdown_report": str(md_path),
        "text_report": str(txt_path)
    }


def _maybe_plot(
    gap: Sequence[float],
    s1: Sequence[float],
    s2: Sequence[float],
    rows_data: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str = "eval",
) -> Dict[str, str]:
    """
    Enhanced visualization with more plots and better styling.
    Returns a dictionary mapping plot names to file paths.
    """
    plot_paths = {}
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[WARN] Matplotlib not available, skipping visualization: {e}")
        return plot_paths

    gap = list(gap)
    s1 = list(s1)
    s2 = list(s2)
    
    # 设置更好的绘图样式
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        pass
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Histogram: Gap
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(gap, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero gap')
        ax.axvline(x=statistics.mean(gap), color='orange', linestyle='--', linewidth=2, label=f'Mean: {statistics.mean(gap):.4f}')
        ax.set_title("Gap Distribution (S1 - S2)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Gap", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = out_dir / f"{prefix}_gap_hist.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_paths['gap_histogram'] = str(plot_file)
        print(f"[PLOT] Saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"[WARN] Gap histogram plot failed: {e}")
    
    # 2. Scatter: S1 vs S2
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        # 根据gap大小着色
        scatter = ax.scatter(s1, s2, c=gap, cmap='RdYlBu', alpha=0.6, s=20, 
                            vmin=-max(abs(min(gap)), abs(max(gap))) if gap else 0, 
                            vmax=max(abs(min(gap)), abs(max(gap))) if gap else 1)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
        ax.set_title("S1 vs S2 (colored by Gap)", fontsize=14, fontweight='bold')
        ax.set_xlabel("S1 (Text1-Text2)", fontsize=12)
        ax.set_ylabel("S2 (Text1-Text3)", fontsize=12)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Gap (S1 - S2)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = out_dir / f"{prefix}_s1_vs_s2_scatter.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_paths['s1_vs_s2_scatter'] = str(plot_file)
        print(f"[PLOT] Saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"[WARN] S1 vs S2 scatter plot failed: {e}")
    
    # 3. Boxplot: S1/S2
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([s1, s2], labels=['S1\n(Text1-Text2)', 'S2\n(Text1-Text3)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title("S1 vs S2 Distribution Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("Similarity Score", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plot_file = out_dir / f"{prefix}_s1_s2_boxplot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_paths['s1_s2_boxplot'] = str(plot_file)
        print(f"[PLOT] Saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"[WARN] Boxplot failed: {e}")
    
    # 4. Violin plot (more detailed distribution)
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        parts = ax.violinplot([s1, s2], positions=[1, 2], showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['S1\n(Text1-Text2)', 'S2\n(Text1-Text3)'])
        ax.set_title("Distribution Density (Violin Plot)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Similarity Score", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plot_file = out_dir / f"{prefix}_violin.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_paths['violin_plot'] = str(plot_file)
        print(f"[PLOT] Saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"[WARN] Violin plot failed: {e}")
    
    # 5. Gap vs GT offset (if GT data available)
    if rows_data and 'gt_offset' in rows_data[0]:
        try:
            gt_offsets = [r['gt_offset'] for r in rows_data]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(gap, gt_offsets, alpha=0.5, s=15)
            ax.set_title("Gap vs Ground Truth Offset", fontsize=14, fontweight='bold')
            ax.set_xlabel("Gap (S1 - S2)", fontsize=12)
            ax.set_ylabel("GT Offset (S2 - GT Score)", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = out_dir / f"{prefix}_gap_vs_gt.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_paths['gap_vs_gt'] = str(plot_file)
            print(f"[PLOT] Saved: {plot_file}")
            plt.close()
        except Exception as e:
            print(f"[WARN] Gap vs GT plot failed: {e}")
    
    # 6. Cumulative distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        import numpy as np
        for data, label, color in zip([s1, s2, gap], ['S1', 'S2', 'Gap'], ['blue', 'red', 'green']):
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, label=label, color=color, linewidth=2)
        ax.set_title("Cumulative Distribution Function", fontsize=14, fontweight='bold')
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = out_dir / f"{prefix}_cdf.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_paths['cumulative_distribution'] = str(plot_file)
        print(f"[PLOT] Saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"[WARN] CDF plot failed: {e}")
    
    # 7. S2 vs GT Score (if GT data available with mapping)
    if rows_data and 'gt_score_mapped' in rows_data[0]:
        try:
            gt_mapped = [r['gt_score_mapped'] for r in rows_data]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(s2, gt_mapped, alpha=0.5, s=15)
            
            # 添加趋势线
            import numpy as np
            z = np.polyfit(s2, gt_mapped, 1)
            p = np.poly1d(z)
            ax.plot(sorted(s2), p(sorted(s2)), "r--", alpha=0.8, label=f'Trend line (r={np.corrcoef(s2, gt_mapped)[0,1]:.3f})')
            
            ax.set_title("S2 vs Mapped GT Score", fontsize=14, fontweight='bold')
            ax.set_xlabel("S2 (Text1-Text3 Similarity)", fontsize=12)
            ax.set_ylabel("Mapped GT Score (0-1)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = out_dir / f"{prefix}_s2_vs_gt.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_paths['s2_vs_gt'] = str(plot_file)
            print(f"[PLOT] Saved: {plot_file}")
            plt.close()
        except Exception as e:
            print(f"[WARN] S2 vs GT plot failed: {e}")
    
    return plot_paths


def _compute_correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """计算两个序列的相关性"""
    try:
        from scipy.stats import pearsonr, spearmanr
        
        x_list = list(x)
        y_list = list(y)
        
        # Pearson correlation
        pearson_corr, _ = pearsonr(x_list, y_list)
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(x_list, y_list)
        
        return {
            "pearson": float(pearson_corr),
            "spearman": float(spearman_corr)
        }
    except Exception:
        return {
            "pearson": 0.0,
            "spearman": 0.0
        }


def evaluate_dataset(
    dataset_rows: Sequence[Dict[str, Any]],
    out_dir: str | Path | None = None,
    similarity_model: str = "WhereIsAI/UAE-Large-V1",
    pooling_strategy: str = "cls",
    device: str = "cuda",
    prefix: str = "eval",
    gt_original_range: Tuple[float, float] = (0, 5),
    gt_target_range: Tuple[float, float] = (0, 1),
    map_gt_scores: bool = True,
) -> Dict[str, Any]:
    """
    Compute:
      S1 = sim(text1, text2)
      S2 = sim(text1, text3)
      Gap = S1 - S2 (larger gap => harder negatives)

    Also:
      - validity: fraction where S2 < S1
      - method contribution: average gap by method presence in methods_used
      - GT offset: compare S2 with dataset 'score' (ground truth similarity label)
    
    Args:
        dataset_rows: List of dictionaries with 'text1', 'text2', 'text3', 'score', 'methods_used', 'id'
        out_dir: Output directory for results
        similarity_model: Model name for similarity computation
        pooling_strategy: Pooling strategy for the model
        device: Device to run model on
        prefix: Prefix for output files
        gt_original_range: Original range of GT scores (min, max)
        gt_target_range: Target range for GT scores (min, max)
        map_gt_scores: Whether to map GT scores to target range
    
    Returns:
        Dictionary with metrics and file paths
    """
    if not dataset_rows:
        raise ValueError("dataset_rows is empty")

    text1_list: List[str] = []
    text2_list: List[str] = []
    text3_list: List[str] = []
    gt_list: List[float] = []
    methods_used_list: List[List[str]] = []
    row_ids: List[str] = []

    for i, row in enumerate(dataset_rows):
        t1 = _normalize_text(row.get("text1", ""))
        t2 = _normalize_text(row.get("text2", ""))
        t3 = _normalize_text(row.get("text3", ""))
        if not t1 or not t2 or not t3:
            continue
        text1_list.append(t1)
        text2_list.append(t2)
        text3_list.append(t3)
        gt_list.append(float(row.get("score", 0.0)))
        methods_used_list.append(list(row.get("methods_used") or []))
        row_ids.append(row.get("id", f"row_{i}"))

    if not text1_list:
        raise ValueError("No valid rows after normalization.")

    similarity_backend = "angle_emb"
    try:
        from angle_emb import AnglE

        angle = AnglE.from_pretrained(similarity_model, pooling_strategy=pooling_strategy)
        if device:
            try:
                angle = angle.cuda()
            except Exception:
                pass

        emb1 = angle.encode(text1_list)
        emb2 = angle.encode(text2_list)
        emb3 = angle.encode(text3_list)

        s1 = _cosine_similarity_batch(emb1, emb2)
        s2 = _cosine_similarity_batch(emb1, emb3)
    except Exception:
        similarity_backend = "bow_cosine"
        s1 = _similarity_bow(text1_list, text2_list)
        s2 = _similarity_bow(text1_list, text3_list)

    # Convert to python lists for pure-Python stats (no numpy dependency).
    s1 = [float(x*5) for x in s1]
    s2 = [float(x*5) for x in s2]

    gap = [a - b for a, b in zip(s1, s2)]  # larger => text3 hurts similarity more

    validity_ratio = sum(1 for a, b in zip(s1, s2) if b < a) / max(1, len(s1))

    # Map GT scores if needed
    gt_mapped_list = []
    if map_gt_scores:
        for gt_score in gt_list:
            mapped = _map_gt_score(gt_score, gt_original_range, gt_target_range)
            gt_mapped_list.append(mapped)
    else:
        gt_mapped_list = gt_list.copy()
    
    # GT offset: compare s2 with mapped GT score
    gt_offset = [b - g for b, g in zip(s2, gt_mapped_list)]

    # Prepare detailed per-row data
    rows_data = []
    for i in range(len(row_ids)):
        row_entry = {
            "id": row_ids[i],
            "text1": text1_list[i],
            "text2": text2_list[i],
            "text3": text3_list[i],
            "s1": s1[i],
            "s2": s2[i],
            "gap": gap[i],
            "gt_score_original": gt_list[i] if i < len(gt_list) else None,
            "gt_score_mapped": gt_mapped_list[i] if i < len(gt_mapped_list) else None,
            "gt_offset": gt_offset[i] if i < len(gt_offset) else None,
            "methods_used": methods_used_list[i] if i < len(methods_used_list) else []
        }
        rows_data.append(row_entry)

    metrics: Dict[str, Any] = {
        "n": int(len(s1)),
        "similarity_backend": similarity_backend,
        "S1_stats": _compute_basic_stats(s1),
        "S2_stats": _compute_basic_stats(s2),
        "Gap_stats": _compute_basic_stats(gap),
        "validity_ratio_S2_lt_S1": validity_ratio,
        "GT_offset_stats": _compute_basic_stats(gt_offset),
    }
    
    # Add GT mapping info
    if map_gt_scores:
        metrics["gt_mapping"] = {
            "original_range": list(gt_original_range),
            "target_range": list(gt_target_range),
            "enabled": True
        }
        metrics["GT_mapped_stats"] = _compute_basic_stats(gt_mapped_list)
        
        # Compute correlation between S2 and mapped GT scores
        metrics["gt_correlation"] = _compute_correlation(s2, gt_mapped_list)

    # Method contribution: average gap over rows where method is present.
    method_contrib: Dict[str, Dict[str, Any]] = {}
    for m in METHOD_NAMES:
        mask = [m in used for used in methods_used_list]
        sel_gap = [float(g) for g, ok in zip(gap, mask) if ok]
        method_contrib[m] = {
            "n": int(len(sel_gap)),
            "gap_stats": _compute_basic_stats(sel_gap),
        }

    metrics["method_contribution"] = method_contrib

    # Find top contributors (highest mean gap).
    best_methods = []
    for m, info in method_contrib.items():
        best_methods.append((info["gap_stats"]["mean"], m, info["n"]))
    best_methods.sort(reverse=True)
    metrics["best_methods_by_mean_gap"] = [
        {"method": m, "mean_gap": mean_gap, "n": n}
        for mean_gap, m, n in best_methods[:5]
        if n > 0
    ]

    # Output files
    paths: Dict[str, str] = {}
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save detailed per-row data (JSON) - directly in out_dir
        detailed_json_path = out_path / f"{prefix}_detailed.json"
        with detailed_json_path.open("w", encoding="utf-8") as f:
            json.dump(rows_data, f, ensure_ascii=False, indent=2)
        paths["detailed_json"] = str(detailed_json_path)
        print(f"[DATA] Saved detailed data: {detailed_json_path}")

        # 2. Save metrics summary (JSON) - directly in out_dir
        json_path = out_path / f"{prefix}_metrics.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        paths["metrics_json"] = str(json_path)
        print(f"[DATA] Saved metrics: {json_path}")

        # 3. Generate formatted text report (Markdown + TXT) - in out_dir
        report_paths = _generate_text_report(metrics, rows_data, out_path, prefix)
        paths.update(report_paths)

        # 4. Generate enhanced visualizations - in out_dir
        plot_paths = _maybe_plot(gap=gap, s1=s1, s2=s2, rows_data=rows_data, out_dir=out_path, prefix=prefix)
        paths.update({f"{k}_plot": v for k, v in plot_paths.items()})
        
        metrics["paths"] = paths

    return metrics