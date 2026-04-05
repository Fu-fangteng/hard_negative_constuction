# Stage 2：NLI 困难负样本构造设计说明

---

## 1. 项目目标

以 nli_for_simcse 的三元组数据 `{anchor, positive, negative}` 为基础，自动化构建高质量、可控的困难负样本（hard negative），扩展为四元组 `{anchor, positive, negative, hard_negative}`，为对比学习等任务提供更具挑战性的训练集。

---

## 2. 总体流程

```
原始数据（parquet / jsonl）
    ↓
数据加载与采样（stage2/data_loader.py）
    ↓  NLIRecord(id, anchor, pos, neg)
特征提取（stage2/feature_extractor.py）
    ├─ regex：numbers / negations / logic_words / degree_words / pronouns
    ├─ regex 实体识别（降级方案，无需 spaCy）
    └─ spaCy（可选）：entities / subject_candidates / object_candidates
    ↓
困难负样本构造（stage2/constructors.py × 10 方法）
    ├─ 每条样本对所有方法独立运行
    └─ 失败返回 None，成功返回 hard_neg
    ↓
结果存储（stage2/builder.py → PipelineRunner）
    ├─ constructed_data.json
    ├─ method_stat.json
    └─ construction_summary.txt
    ↓
汇总分析（stage2/analyzer.py）
    └─ difference.md（各方法对比）
```

---

## 3. 目录结构

```
hard_neg/
├── stage2/                     # 核心代码
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── constructors.py
│   ├── builder.py
│   ├── analyzer.py
│   └── run_stage2.py
└── data/stage2/
    ├── preprocessed/           # 预处理后数据（不入库）
    └── processed/
        └── <method_name>/
            └── Regular/
                ├── constructed_data.json   # 构造结果（不入库）
                ├── method_stat.json        # 统计文件
                ├── construction_log.jsonl  # 逐条日志（不入库）
                └── construction_summary.txt
```

---

## 4. 关键设计决策

### 4.1 每方法独立运行（vs Stage 1 的 first-method-wins）

Stage 1 使用 auto 模式：按优先级尝试方法，成功即停止。结果导致 `direct_negation_attack`（第 4 优先级）抢占 59% 的样本，后续方法几乎没有机会。

Stage 2 改为**每方法独立运行**：每条样本对所有 10 种方法各跑一遍，独立记录成功/失败。这样：
- 可以精确统计每种方法的真实成功率
- 便于后续按最优方法分配（而非 first-method-wins）

### 4.2 实体识别降级策略

Stage 1 的 `entities` 字段完全依赖 spaCy，spaCy 未安装时始终为空，导致 `entity_pronoun_substitution` 等方法在无 spaCy 时完全失效。

Stage 2 实现 regex 降级方案（`_extract_entities_regex`）：
1. 称谓+姓名模式（`Dr. Smith`）
2. 大写词序列（过滤 `_COMMON_CAPS` 词表）
3. 全大写缩写（`NASA`、`WHO`）

无 spaCy 时也能识别 50%+ 的实体。

### 4.3 真实实体替换表

Stage 1 的兜底策略是 "another X"（如 "another John"），语义扰动效果差。Stage 2 引入 ~80 条目的真实替换表：

- 人名：John↔Michael, Mary↔Sarah, David↔James, ...
- 地名：Paris↔London, Tokyo↔Berlin, New York↔Los Angeles, ...
- 机构：Google↔Apple, Harvard↔Yale, NASA↔ESA, ...

---

## 5. 10 种构造方法概览

| 方法 | Stage 2 成功率 | spaCy 必须 |
|---|---|---|
| premise_disruption | 100% | ❌ |
| direct_negation_attack | 91.9% | ❌ |
| role_swap | 66.4% | ✅ |
| entity_pronoun_substitution | 20.3% | 可选 |
| double_negation_attack | 8.1% | ❌ |
| scope_degree_scaling | 7.5% | ❌ |
| logical_operator_rewrite | 5.2% | ❌ |
| concept_hierarchy_shift | 3.9% | 可选 |
| temporal_causal_inversion | 1.9% | ❌ |
| numeric_metric_transform | 1.6% | ❌ |

> 详细方法说明见 [hard_neg_construction_method.md](hard_neg_construction_method.md)

---

## 6. 运行命令

```bash
# 全量运行（推荐）
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 1000 \
    --methods all \
    --recognizer regular

# 单方法调试
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 100 \
    --methods role_swap \
    --recognizer regular
```

---

## 7. 后续扩展建议

- **自动方法选择**：根据特征分配最优方法，替代当前的 first-method-wins 聚合逻辑
- **扩充 `concept_hierarchy_shift` 词表**：接入 WordNet hypernym/hyponym，将成功率从 3.9% 提升至 20%+
- **LLM 路径**：`builder.py` 已预留 `llm_engine` 参数，可接入 Qwen 等本地模型
- **angle_emb 评估**：Stage 2 尚未运行评估，可复用 `stage1/evaluator.py` 对 hard_neg 质量打分
