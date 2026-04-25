# Phase 2 训练改造说明：含筛选机制的完整流程

---

## 整体流程

```
原始构造数据（constructed_data.json）
        ↓
  加载 Phase 1 checkpoint
        ↓
  计算每条样本的 T 和 T*
        ↓
  按筛选规则过滤（保留情况 B 和 C）
        ↓
  保存高质量数据（含 T、T* 字段）
        ↓
  用高质量数据 × 4 种 Loss 分别微调
        ↓
  汇总对比结果
```

---

## 第一步：计算 T 和 T*

### 函数：`compute_triplet_losses`

在 `load_from_json` 之后，新增一个函数，用 Phase 1 的模型对所有样本计算 T 和 T*：

```python
import torch
import torch.nn.functional as F

def compute_triplet_losses(
    model: SentenceTransformer,
    dataset: Dataset,
    margin: float = 0.2,
    batch_size: int = 64,
) -> Dataset:
    """
    对数据集中每条样本计算：
      T  = TripletLoss(anchor, pos, neg_orig)
      T* = TripletLoss(anchor, pos, hard_neg)
    结果以新字段写回 Dataset。

    输入 Dataset 必须包含字段：anchor, positive, neg_orig, negative(=hard_neg)
    输出 Dataset 新增字段：T, T_star, T_star_minus_T
    """
    model.eval()
    all_T, all_T_star = [], []

    anchors   = dataset["anchor"]
    positives = dataset["positive"]
    neg_origs = dataset["neg_orig"]
    hard_negs = dataset["negative"]

    with torch.no_grad():
        for i in range(0, len(anchors), batch_size):
            batch_a   = anchors[i : i + batch_size]
            batch_p   = positives[i : i + batch_size]
            batch_n   = neg_origs[i : i + batch_size]
            batch_hn  = hard_negs[i : i + batch_size]

            # 编码（normalize 到单位球，cos 距离 = 1 - cos_sim）
            emb_a  = model.encode(batch_a,  convert_to_tensor=True, normalize_embeddings=True)
            emb_p  = model.encode(batch_p,  convert_to_tensor=True, normalize_embeddings=True)
            emb_n  = model.encode(batch_n,  convert_to_tensor=True, normalize_embeddings=True)
            emb_hn = model.encode(batch_hn, convert_to_tensor=True, normalize_embeddings=True)

            # cosine distance = 1 - cosine_similarity
            d_ap  = 1 - F.cosine_similarity(emb_a, emb_p)
            d_an  = 1 - F.cosine_similarity(emb_a, emb_n)
            d_ahn = 1 - F.cosine_similarity(emb_a, emb_hn)

            # Triplet Loss = max(0, d(a,p) - d(a,n) + margin)
            T      = torch.clamp(d_ap - d_an  + margin, min=0).cpu().tolist()
            T_star = torch.clamp(d_ap - d_ahn + margin, min=0).cpu().tolist()

            all_T.extend(T)
            all_T_star.extend(T_star)

    delta = [t_star - t for t_star, t in zip(all_T_star, all_T)]

    return dataset.add_column("T", all_T) \
                  .add_column("T_star", all_T_star) \
                  .add_column("T_star_minus_T", delta)
```

---

## 第二步：筛选高质量样本

### 函数：`filter_hard_negatives`

根据 T 和 T* 的五种情况，只保留情况 B 和 C：

```python
def filter_hard_negatives(
    dataset: Dataset,
    t_star_max: float = 2.0,    # T* 上限：超过此值疑似构造错误，丢弃
) -> tuple[Dataset, dict]:
    """
    筛选规则：
      情况 B：T == 0 且 T* > 0              → 保留（最优质）
      情况 C：T > 0 且 T* > 0 且 T*-T > 0  → 保留（有增量价值）
      其余情况（A / D / E）                  → 丢弃
      额外：T* > t_star_max                  → 丢弃（疑似构造错误）

    返回：(筛选后的 Dataset, 统计信息 dict)
    """
    total = len(dataset)
    stats = {"total": total, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
             "overflow": 0, "kept": 0}

    def classify(row):
        T      = row["T"]
        T_star = row["T_star"]
        delta  = row["T_star_minus_T"]
        eps    = 1e-6   # 浮点容差

        if T_star > t_star_max:
            return "overflow"
        if T < eps and T_star < eps:
            return "A"
        if T < eps and T_star > eps:
            return "B"
        if T > eps and T_star > eps and delta > eps:
            return "C"
        if T > eps and T_star > eps and delta <= eps:
            return "D"
        if T > eps and T_star < eps:
            return "E"
        return "A"   # 兜底

    keep_rows = []
    for row in dataset:
        case = classify(row)
        stats[case] = stats.get(case, 0) + 1
        if case in ("B", "C"):
            keep_rows.append({**row, "case": case})

    stats["kept"] = len(keep_rows)
    stats["kept_ratio"] = round(stats["kept"] / total, 4) if total else 0

    filtered_ds = Dataset.from_list(keep_rows)
    return filtered_ds, stats
```

---

## 第三步：保存高质量数据

### 函数：`save_filtered_dataset`

把筛选结果（含 T、T*、case 字段）保存到磁盘，供后续训练使用，也方便复盘分析：

```python
import json
from pathlib import Path

def save_filtered_dataset(
    dataset: Dataset,
    stats: dict,
    save_dir: Path,
    exp_name: str,
) -> Path:
    """
    保存筛选后的高质量数据：
      - filtered_data.json   完整数据（含 T, T_star, T_star_minus_T, case）
      - filter_stats.json    筛选统计（各情况数量、保留率）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path  = save_dir / f"filtered_data_{exp_name}.json"
    stats_path = save_dir / f"filter_stats_{exp_name}.json"

    # 保存数据
    records = [row for row in dataset]
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 保存统计
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"  Filtered data saved → {data_path}")
    logger.info(f"  Filter stats: {stats}")
    return data_path
```

---

## 第四步：从筛选数据构建训练集

根据 Loss 类型，从筛选后的数据构建对应格式的训练集：

```python
def build_train_datasets(
    filtered_ds: Dataset,
    loss_type: str,
) -> tuple:
    """
    根据 loss_type 返回 (train_datasets, losses_dict, batch_sampler)。

    triplet_cascade     → 两份显式三元组 {base, hard}
    batch_hard          → (sentence, label) 格式
    batch_semi_hard     → (sentence, label) 格式
    batch_hard_soft_margin → (sentence, label) 格式
    """
    from sentence_transformers.losses import (
        TripletLoss,
        BatchHardTripletLoss,
        BatchSemiHardTripletLoss,
        BatchHardSoftMarginTripletLoss,
    )
    from sentence_transformers.training_args import BatchSamplers

    if loss_type == "triplet_cascade":
        ds_base = Dataset.from_dict({
            "anchor":   filtered_ds["anchor"],
            "positive": filtered_ds["positive"],
            "negative": filtered_ds["neg_orig"],
        })
        ds_hard = Dataset.from_dict({
            "anchor":   filtered_ds["anchor"],
            "positive": filtered_ds["positive"],
            "negative": filtered_ds["negative"],   # hard_neg
        })
        train_datasets = {"base": ds_base, "hard": ds_hard}
        losses = {
            "base": TripletLoss(margin=0.2),   # model 在 run_phase2 内绑定
            "hard": TripletLoss(margin=0.2),
        }
        batch_sampler = BatchSamplers.NO_DUPLICATES

    else:
        # BatchHard* 系列：转换为 (sentence, label) 格式
        sentences, labels = [], []
        for i, row in enumerate(filtered_ds):
            base_label = i * 2
            sentences += [row["anchor"], row["positive"], row["negative"]]
            labels    += [base_label, base_label, base_label + 1]

        label_ds = Dataset.from_dict({"sentence": sentences, "label": labels})
        train_datasets = label_ds
        batch_sampler  = BatchSamplers.GROUP_BY_LABEL

        if loss_type == "batch_hard":
            losses = BatchHardTripletLoss(margin=0.5)
        elif loss_type == "batch_semi_hard":
            losses = BatchSemiHardTripletLoss(margin=0.5)
        elif loss_type == "batch_hard_soft_margin":
            losses = BatchHardSoftMarginTripletLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    return train_datasets, losses, batch_sampler
```

> **注意**：`TripletLoss` 的 `model` 参数在 `run_phase2` 内绑定，`build_train_datasets` 只负责返回 loss 配置，实际的 `loss(model)` 调用在 `run_phase2` 里完成。

---

## 第五步：改造 run_phase2

接收筛选后的 `filtered_ds`，按 `loss_name` 独立训练：

```python
def run_phase2(
    name: str,
    loss_name: str,
    model_path: str,               # Phase 1 final checkpoint 路径
    filtered_ds: Dataset,          # 筛选后的高质量数据（含 T, T_star 字段）
    eval_orig_ds: Dataset,
    eval_hard_ds: Dataset,
    phase1_results: dict,
) -> dict:
    run_label  = f"{RUN_ID}_{name}_phase2_{loss_name}"
    output_dir = Path(f"models/{RUN_ID}_compare/{name}/phase2_{loss_name}")
    final_dir  = output_dir / "final"
    log_dir    = output_dir / "logs"
    for d in [output_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_file  = log_dir / f"metrics_{run_label}.csv"
    json_file = log_dir / f"metrics_{run_label}.json"

    # 每种 Loss 从同一 Phase 1 checkpoint 独立加载，保证对比公平
    model = SentenceTransformer(model_path)

    # test / debug 截断（在筛选后的数据上截断）
    train_ds = filtered_ds
    if TEST_MODE:
        train_ds = train_ds.select(range(min(TEST_SIZE, len(train_ds))))
    elif DEBUG:
        train_ds = train_ds.select(range(min(DEBUG_SIZE, len(train_ds))))

    logger.info(f"  [Phase2/{loss_name}] train samples (after filter): {len(train_ds)}")
    logger.info(f"  [Phase2/{loss_name}] T mean={sum(train_ds['T'])/len(train_ds):.4f}, "
                f"T* mean={sum(train_ds['T_star'])/len(train_ds):.4f}")

    # 构建训练集和 Loss
    train_datasets, losses, batch_sampler = build_train_datasets(train_ds, loss_name)

    # 对 TripletLoss 绑定 model
    if loss_name == "triplet_cascade":
        from sentence_transformers.losses import TripletLoss
        losses = {
            "base": TripletLoss(model, triplet_margin=0.2),
            "hard": TripletLoss(model, triplet_margin=0.2),
        }
    else:
        # BatchHard* 系列直接传 model
        from sentence_transformers.losses import (
            BatchHardTripletLoss, BatchSemiHardTripletLoss,
            BatchHardSoftMarginTripletLoss,
        )
        if loss_name == "batch_hard":
            losses = BatchHardTripletLoss(model, margin=0.5)
        elif loss_name == "batch_semi_hard":
            losses = BatchSemiHardTripletLoss(model, margin=0.5)
        elif loss_name == "batch_hard_soft_margin":
            losses = BatchHardSoftMarginTripletLoss(model)

    evaluator_orig = TripletEvaluator(
        anchors=eval_orig_ds["anchor"],
        positives=eval_orig_ds["positive"],
        negatives=eval_orig_ds["negative"],
        name=f"{name}-p2-{loss_name}-orig",
    )
    evaluator_hard = TripletEvaluator(
        anchors=eval_hard_ds["anchor"],
        positives=eval_hard_ds["positive"],
        negatives=eval_hard_ds["negative"],
        name=f"{name}-p2-{loss_name}-hard",
    )

    metrics_callback = MetricsLoggerCallback(csv_file, json_file)

    eval_steps = 100 if not TEST_MODE else 20
    save_steps = 100 if not TEST_MODE else 20

    args = SentenceTransformerTrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = 3,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-6,       # Phase 1 的 1/10
        warmup_ratio                = 0.1,
        fp16=True, bf16=False,
        batch_sampler               = batch_sampler,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        logging_steps               = 20,
        run_name                    = run_label,
    )

    from transformers import EarlyStoppingCallback
    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = args,
        train_dataset = train_datasets,
        eval_dataset  = eval_hard_ds,
        loss          = losses,
        evaluator     = evaluator_orig,    # orig neg 不退化作为保存标准
        callbacks     = [
            metrics_callback,
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    trainer.train()
    model.save_pretrained(str(final_dir))

    p2_orig = evaluator_orig(model)
    p2_hard = evaluator_hard(model)

    def acc(d): return get_accuracy(d)
    logger.info(f"  [Phase2/{loss_name}] orig: "
                f"{acc(phase1_results['p1_orig']):.4f} → {acc(p2_orig):.4f}  "
                f"(Δ {acc(p2_orig) - acc(phase1_results['p1_orig']):+.4f})")
    logger.info(f"  [Phase2/{loss_name}] hard: "
                f"{acc(phase1_results['p1_hard']):.4f} → {acc(p2_hard):.4f}  "
                f"(Δ {acc(p2_hard) - acc(phase1_results['p1_hard']):+.4f})")

    return {
        "exp_name":      name,
        "loss_name":     loss_name,
        "train_size":    len(train_ds),
        "p1_orig":       acc(phase1_results["p1_orig"]),
        "p1_hard":       acc(phase1_results["p1_hard"]),
        "p2_orig":       acc(p2_orig),
        "p2_hard":       acc(p2_hard),
        "delta_p2_orig": acc(p2_orig) - acc(phase1_results["p1_orig"]),
        "delta_p2_hard": acc(p2_hard) - acc(phase1_results["p1_hard"]),
    }
```

---

## 第六步：改造主循环

```python
PHASE2_LOSSES = [
    "triplet_cascade",
    "batch_hard",
    "batch_semi_hard",
    "batch_hard_soft_margin",
]

FILTER_SAVE_DIR = Path(f"models/{RUN_ID}_compare/filtered_data")

all_summaries: list[dict] = []

for exp_name, exp_paths in DATASETS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"[EXPERIMENT] {exp_name.upper()}")
    logger.info(f"{'='*60}")

    # ── Phase 1 ──────────────────────────────────────────────
    p1_model, p1_results = run_phase1(
        name         = exp_name,
        data_paths   = exp_paths,
        eval_orig_ds = eval_orig,
        eval_hard_ds = eval_hard,
    )
    p1_model_path = str(
        Path(f"models/{RUN_ID}_compare/{exp_name}/phase1/final")
    )

    # ── 筛选：计算 T / T*，过滤高质量样本 ───────────────────
    logger.info(f"\n  [Filter] Computing T and T* with Phase 1 model ...")
    full_ds = load_from_json(exp_paths)
    full_ds = full_ds.filter(lambda x: x["id"] not in shared_test_ids)

    # 计算 T 和 T*
    full_ds_with_loss = compute_triplet_losses(p1_model, full_ds)

    # 筛选情况 B 和 C
    filtered_ds, filter_stats = filter_hard_negatives(full_ds_with_loss)
    logger.info(f"  [Filter] {exp_name}: {filter_stats['total']} → "
                f"{filter_stats['kept']} kept ({filter_stats['kept_ratio']:.1%})")
    logger.info(f"  [Filter] Case breakdown: "
                f"A={filter_stats['A']}, B={filter_stats['B']}, "
                f"C={filter_stats['C']}, D={filter_stats['D']}, "
                f"E={filter_stats['E']}, overflow={filter_stats['overflow']}")

    # 保存筛选结果（含 T, T_star 字段）
    save_filtered_dataset(filtered_ds, filter_stats, FILTER_SAVE_DIR, exp_name)

    # ── Phase 2：每种 Loss 各跑一次 ──────────────────────────
    for loss_name in PHASE2_LOSSES:
        logger.info(f"\n  --- Phase 2 / {loss_name} ---")
        summary = run_phase2(
            name           = exp_name,
            loss_name      = loss_name,
            model_path     = p1_model_path,
            filtered_ds    = filtered_ds,
            eval_orig_ds   = eval_orig,
            eval_hard_ds   = eval_hard,
            phase1_results = p1_results,
        )
        all_summaries.append(summary)
```

---

## 输出目录结构

```
models/{RUN_ID}_compare/
├── filtered_data/                        ← 筛选结果（所有实验共用目录）
│   ├── filtered_data_llm.json            ← 含 T, T_star, T_star_minus_T, case 字段
│   ├── filter_stats_llm.json             ← 筛选统计
│   ├── filtered_data_regular.json
│   ├── filter_stats_regular.json
│   ├── filtered_data_combined.json
│   └── filter_stats_combined.json
├── llm/
│   ├── phase1/
│   │   ├── final/                        ← Phase 1 checkpoint，所有 Phase 2 共享
│   │   └── logs/
│   ├── phase2_triplet_cascade/
│   │   ├── final/
│   │   └── logs/
│   ├── phase2_batch_hard/
│   ├── phase2_batch_semi_hard/
│   └── phase2_batch_hard_soft_margin/
├── regular/
│   └── ...（同上）
└── combined/
    └── ...（同上）
```

---

## 筛选数据的字段说明

`filtered_data_{exp_name}.json` 中每条记录包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 原始样本 ID |
| `anchor` | str | Anchor 句子 |
| `positive` | str | 正样本 |
| `neg_orig` | str | 原始负样本 |
| `negative` | str | 构造的 hard_neg |
| `method` | str | 构造方法名 |
| `recognizer` | str | Regular / LLM |
| `T` | float | 原始三元组的 TripletLoss 值 |
| `T_star` | float | hard_neg 三元组的 TripletLoss 值 |
| `T_star_minus_T` | float | T* - T，正值代表 hard_neg 更难 |
| `case` | str | B 或 C |

---

## 运行方式

```bash
# test 模式（小数据，验证完整流程）
python train_hard_neg.py --test

# debug 模式（只验证代码不报错）
python train_hard_neg.py --debug

# 正式训练
python train_hard_neg.py
```

---

## 注意事项

**筛选在 Phase 1 结束后、Phase 2 开始前执行一次**，所有 Phase 2 的 Loss 变体共享同一份筛选结果，保证对比的数据基础一致。

**`compute_triplet_losses` 使用 `model.eval()` + `torch.no_grad()`**，不会影响 Phase 1 模型的权重，可以安全地在 Phase 1 结束后立刻调用。

**`t_star_max` 默认值为 2.0**（cosine distance 空间的理论最大值为 2.0），可以根据实际数据分布调整，建议先用 `filter_stats` 查看 overflow 数量再决定是否收紧。

**`BatchHard*` 系列必须使用 `GROUP_BY_LABEL` 作为 batch_sampler**，`TripletLoss` 系列使用 `NO_DUPLICATES`，两者不能混用。
