# STS/NLI 困难负样本构造（Hard Negatives）

本项目分两个阶段，从 STS / NLI 数据集自动构造"困难负样本（Hard Negatives）"，用于对比学习等任务的训练集增强。

---

## 项目结构

```
hard_neg/
├── stage1/               # Stage 1: STS 数据集困难负样本构造
│   ├── data_utils.py     # 数据加载（STSRecord）
│   ├── sampler.py        # top-k 正样本筛选
│   ├── llm_engine.py     # 本地 LLM 封装（可选）
│   ├── prompts.py        # LLM prompt 模板
│   ├── formatter.py      # 特征提取（regex + spaCy + LLM）
│   ├── constructors.py   # 10 种构造方法
│   ├── main_generator.py # 调度 text3 生成
│   ├── evaluator.py      # 相似度评估（angle_emb）
│   └── run_pipeline.py   # 端到端脚本
├── stage2/               # Stage 2: NLI 三元组 → 四元组
│   ├── data_loader.py    # 加载 parquet/jsonl（NLIRecord）
│   ├── feature_extractor.py  # 特征提取（regex + spaCy，可降级）
│   ├── constructors.py   # 复用 stage1 方法 + 增强实体替换表
│   ├── builder.py        # PipelineRunner（per-method 独立运行）
│   ├── analyzer.py       # 汇总统计、difference.md 生成
│   └── run_stage2.py     # 主运行脚本
├── data/
│   ├── raw/              # 原始输入（parquet/jsonl，不入库）
│   ├── stage1/           # Stage 1 输出
│   └── stage2/
│       ├── preprocessed/ # 预处理后数据（不入库）
│       └── processed/    # 各方法构造结果
├── tests/
│   ├── conftest.py
│   ├── stage1/           # Stage 1 单元测试
│   └── stage2/           # Stage 2 单元测试（58 个，全部通过）
├── docs/                 # 项目文档
└── configs/              # 配置文件
```

---

## Stage 1：STS 困难负样本构造

### 数据格式

- 输入：`(text1, text2, score)` — STS 正样本对
- 输出：`(text1, text2, text3)` — text3 为构造的困难负样本

### 运行

```bash
# 基本运行（100 条，auto 方法选择）
python stage1/run_pipeline.py --input data/raw/test.jsonl --out_dir data/stage1/run_1 --k 100

# 启用评估（需安装 angle-emb）
python stage1/run_pipeline.py --input data/raw/test.jsonl --out_dir data/stage1/run_1 --k 100 --evaluate

# 指定方法
python stage1/run_pipeline.py --input data/raw/test.jsonl --out_dir data/stage1/run_1 --k 100 \
  --methods direct_negation_attack,entity_pronoun_substitution
```

### 输出文件

| 文件 | 说明 |
|---|---|
| `topk_positives.json/.csv` | 前 k 条正样本对 |
| `formatted_data.json` | 特征 + 方法可用性 |
| `methods_stat.json` | 方法可用性统计 |
| `final_dataset.jsonl/.csv` | 最终困难负样本数据集 |
| `evaluation_report.json` | 评估指标（可选） |

---

## Stage 2：NLI 四元组构造

### 数据格式

- 输入：`(anchor, pos, neg)` — NLI 三元组（parquet 或 jsonl）
- 输出：`(anchor, pos, neg, hard_neg)` — 四元组

### 运行

```bash
# 全方法独立运行（推荐）
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 1000 \
    --methods all \
    --recognizer regular

# 单方法
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 1000 \
    --methods role_swap \
    --recognizer regular
```

### 输出目录结构

```
data/stage2/processed/
└── <method_name>/
    └── Regular/
        ├── constructed_data.json   # {id, anchor, pos, neg, hard_neg, success, ...}
        ├── method_stat.json        # 成功率、失败原因统计
        ├── construction_log.jsonl  # 逐条构造日志（不入库）
        └── construction_summary.txt
```

---

## 10 种构造方法

| # | 方法 | 类别 | 关键特征 |
|---|---|---|---|
| 1 | `numeric_metric_transform` | 局部事实置换 | 数字、度量单位 |
| 2 | `entity_pronoun_substitution` | 局部事实置换 | 实体名、代词 |
| 3 | `scope_degree_scaling` | 局部事实置换 | 程度副词、量词 |
| 4 | `direct_negation_attack` | 极性与逻辑反转 | 助动词、谓语 |
| 5 | `double_negation_attack` | 极性与逻辑反转 | 否定词 |
| 6 | `logical_operator_rewrite` | 极性与逻辑反转 | 逻辑连接词 |
| 7 | `role_swap` | 结构与时序重组 | nsubj/dobj（需 spaCy） |
| 8 | `temporal_causal_inversion` | 结构与时序重组 | 时序词、因果词 |
| 9 | `concept_hierarchy_shift` | 知识与常识偏置 | 概念词（内置词表） |
| 10 | `premise_disruption` | 知识与常识偏置 | 通用（兜底方法） |

> 详见 [docs/hard_neg_construction_method.md](docs/hard_neg_construction_method.md)

---

## 依赖与环境

```bash
# conda rl 环境
conda activate rl

# 核心依赖
pip install pandas pyarrow angle-emb spacy
python -m spacy download en_core_web_sm

# 测试
python -m pytest tests/ -v
```

> **注**：spaCy `en_core_web_sm` 可选，未安装时 `role_swap` 等依赖依存句法的方法会失败，其余方法正常运行。

---

## 性能指标（Stage 1，99 samples，angle_emb）

| 指标 | 值 |
|---|---|
| Gap 均值（S1-S2） | **0.712** |
| 有效率（S2 < S1） | **97.98%** |
| `direct_negation_attack` Gap 均值 | 1.045 |

> Stage 2 各方法独立成功率：`role_swap` 66.4%、`direct_negation_attack` 91.9%、`entity_pronoun_substitution` 20.3%  
> 详见 [docs/stage1_vs_stage2_comparison.md](docs/stage1_vs_stage2_comparison.md)
