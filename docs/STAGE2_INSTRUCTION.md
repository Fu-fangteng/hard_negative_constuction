# Stage 2：NLI 四元组困难负样本构造 Pipeline

---

## 一、项目概述

以 nli_for_simcse 的三元组数据 `{anchor, pos, neg}` 为输入，通过自动化构造困难负样本，生成四元组 `{anchor, pos, neg, hard_neg}`，为对比学习等任务提供更具挑战性的训练集。

**核心设计**：
- 每条样本对所有方法**独立运行**（不互相抢占），便于统计和对比
- 支持 Regular（规则 + spaCy）识别方式
- 特征提取可降级：spaCy 未安装时自动使用纯 regex 方案

---

## 二、代码结构

```
hard_neg/
├── stage2/                      # Stage 2 核心代码
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载（NLIRecord，支持 parquet/jsonl）
│   ├── feature_extractor.py     # 特征提取（regex + spaCy，可降级）
│   ├── constructors.py          # 10 种构造方法（复用 stage1 + 增强实体替换表）
│   ├── builder.py               # PipelineRunner（per-method 独立运行）
│   ├── analyzer.py              # 汇总统计、difference.md 生成
│   └── run_stage2.py            # 主运行脚本
├── data/
│   ├── raw/                     # 原始输入（parquet/jsonl，不入库）
│   └── stage2/
│       ├── preprocessed/        # 预处理后数据（不入库）
│       └── processed/           # 各方法构造结果
│           └── <method_name>/
│               └── Regular/
│                   ├── constructed_data.json
│                   ├── method_stat.json
│                   ├── construction_log.jsonl  # 不入库
│                   └── construction_summary.txt
└── tests/stage2/                # 单元测试
```

---

## 三、数据格式规范

### 3.1 原始数据格式

支持 parquet（nli_for_simcse 官方格式）和 jsonl：

```json
{"anchor": "...", "positive": "...", "negative": "..."}
```

### 3.2 NLIRecord（内部数据结构）

```python
@dataclass
class NLIRecord:
    id: str
    anchor: str
    pos: str
    neg: str
```

### 3.3 构造后数据格式（`constructed_data.json`）

```json
[
  {
    "id": "0",
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": "...",
    "method": "direct_negation_attack",
    "recognizer": "Regular",
    "success": true,
    "failure_reason": null
  },
  {
    "id": "1",
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": null,
    "method": "numeric_metric_transform",
    "recognizer": "Regular",
    "success": false,
    "failure_reason": "no_feature_found"
  }
]
```

### 3.4 统计文件格式（`method_stat.json`）

```json
{
  "method_name": "direct_negation_attack",
  "recognizer_type": "Regular",
  "total_samples": 1000,
  "success_count": 919,
  "success_ratio": 0.919,
  "failure_reasons": {
    "no_feature_found": 81,
    "output_same_as_input": 0,
    "empty_output": 0,
    "exception": 0
  }
}
```

---

## 四、核心模块说明

### 4.1 数据加载（`stage2/data_loader.py`）

```python
from stage2.data_loader import load_data, save_preprocessed

records = load_data("data/raw/train-00000-of-00001.parquet", sample_size=1000)
# 或
records = load_data("data/raw/nli_train.jsonl", sample_size=1000)
```

- 自动检测文件格式（parquet / jsonl）
- `sample_size=None` 表示加载全部
- 随机种子固定为 42，保证可复现

### 4.2 特征提取（`stage2/feature_extractor.py`）

```python
from stage2.feature_extractor import extract_regular

features = extract_regular("Michael went to Paris and met with John.")
# features = {
#   "entities": ["Michael", "Paris", "John"],
#   "numbers": [],
#   "logic_words": [],
#   "negations": [],
#   "pronouns": [],
#   "degree_words": [],
#   "subject_candidates": ["Michael"],   # 需 spaCy
#   "object_candidates": ["John"],        # 需 spaCy
# }
```

**实体识别策略**（spaCy 未安装时的 regex 降级方案）：
1. 称谓+姓名模式：`Dr. Smith`、`Mr. Johnson`
2. 大写词序列：非冠词/代词/连词（`_COMMON_CAPS` 过滤）
3. 全大写缩写：`NASA`、`WHO`（2-5 字母，无数字）

### 4.3 构造调度（`stage2/builder.py`）

```python
from stage2.builder import PipelineRunner

runner = PipelineRunner(
    records=records,
    method_name="role_swap",
    recognizer_type="Regular",
    output_dir="data/stage2/processed/role_swap/Regular"
)
result = runner.run()
# result.stats = {"total": 1000, "success": 664, "success_ratio": 0.664, ...}
```

### 4.4 结果分析（`stage2/analyzer.py`）

汇总所有方法的统计数据，生成 `difference.md` 对比报告。

---

## 五、运行方式

### 5.1 全方法运行（推荐）

```bash
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 1000 \
    --methods all \
    --recognizer regular
```

### 5.2 指定方法运行

```bash
python stage2/run_stage2.py \
    --input_path data/raw/train-00000-of-00001.parquet \
    --output_base data/stage2 \
    --sample_size 1000 \
    --methods role_swap,entity_pronoun_substitution \
    --recognizer regular
```

### 5.3 参数说明

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--input_path` | str | 必需 | 输入数据路径（parquet/jsonl） |
| `--output_base` | str | `data/stage2` | 输出根目录 |
| `--sample_size` | int | None（全部） | 采样数量 |
| `--methods` | str | `all` | 逗号分隔方法名，或 `all` |
| `--recognizer` | str | `regular` | `regular`（当前仅支持此选项） |
| `--seed` | int | 42 | 随机种子 |

---

## 六、Stage 2 性能数据（1000 samples，含 spaCy）

| 方法 | 成功率 | 特征覆盖率 |
|---|---|---|
| premise_disruption | **100%** | 100% |
| direct_negation_attack | **91.9%** | 93.9% |
| role_swap | **66.4%** | 67.0%（需 spaCy） |
| entity_pronoun_substitution | 20.3% | 52.9% |
| scope_degree_scaling | 7.5% | 7.4% |
| double_negation_attack | 8.1% | 6.1% |
| logical_operator_rewrite | 5.2% | 5.1% |
| concept_hierarchy_shift | 3.9% | 41.0% |
| temporal_causal_inversion | 1.9% | 1.8% |
| numeric_metric_transform | 1.6% | 2.3% |

> 详见 [stage1_vs_stage2_comparison.md](stage1_vs_stage2_comparison.md)

---

## 七、测试

```bash
# 全部测试（58 个，全部通过）
python -m pytest tests/stage2/ -v

# 特定模块
python -m pytest tests/stage2/test_feature_extractor.py -v
python -m pytest tests/stage2/test_builder.py -v
```

---

## 八、与 Stage 1 的关键差异

| 维度 | Stage 1 | Stage 2 |
|---|---|---|
| 数据源 | STS（text1, text2, score） | NLI（anchor, pos, neg） |
| 样本数 | 100 | 1000 |
| 方法调度 | auto 模式（first-method-wins） | 每方法独立运行，不互抢 |
| 特征提取 | formatter.py | feature_extractor.py |
| 实体识别 | spaCy（无则为空） | regex 降级 + spaCy |
| 实体替换 | "another X" 兜底 | 真实替换表（~80 条目） |
