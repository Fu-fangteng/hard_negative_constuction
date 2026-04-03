# Stage 2 Hard Negative Pipeline — Design Spec

**Date**: 2026-04-02  
**Status**: Approved  
**Approach**: Thin adapter layer (src_v2/) over existing src/

---

## 1. 目标

以 `nli_for_simcse` parquet 数据（`anchor / positive / negative`，274,951 行）为输入，构造四元组训练集 `{anchor, pos, neg, hard_neg}`。

核心创新点：对每种构造方法分别运行 **Regular**（正则）和 **LLM**（Qwen）两条特征识别路径，独立记录输出，最终生成对比报告 `difference.md`。

本阶段 LLM 仅用于**特征提取**，构造逻辑继续使用规则方法（方案 A）。

---

## 2. 数据来源

| 字段 | 说明 |
|---|---|
| `anchor` | 原始句子 |
| `positive` → `pos` | 被扰动对象，生成 `hard_neg` |
| `negative` → `neg` | 原始负样本，保留至输出 |
| `id` | 无，自动生成 `sample_000001` |

开发阶段使用 `sample_size=1000`，随机种子 `seed=42`。

---

## 3. 目录结构

### 代码

```
hard_neg/
├── src/                          # Stage 1，零修改
├── src_v2/
│   ├── __init__.py
│   ├── data_loader.py            # NLIRecord + parquet 读取
│   ├── feature_extractor.py      # Regular / LLM 分流提取
│   ├── builder.py                # 单方法×单识别器 run
│   └── analyzer.py               # 聚合统计 + difference.md
└── scripts_v2/
    └── run_stage2.py             # 入口脚本
```

### 输出

```
Data/
├── preprocessed_data/
│   └── preprocessed_data.json
└── processed_data/
    ├── {method_name}/
    │   ├── Regular/
    │   │   ├── constructed_data.json
    │   │   ├── method_stat.json
    │   │   ├── construction_log.jsonl
    │   │   └── construction_summary.txt
    │   ├── LLM/
    │   │   ├── constructed_data.json
    │   │   ├── method_stat.json
    │   │   ├── construction_log.jsonl
    │   │   └── construction_summary.txt
    │   └── difference.md
    ├── ...（10 种方法各自目录）
    ├── dataset_methods_stat.json
    └── final_dataset.jsonl
```

---

## 4. 数据格式规范

### 4.1 preprocessed_data.json

```json
[
  {"id": "sample_000001", "anchor": "...", "pos": "...", "neg": "..."}
]
```

### 4.2 constructed_data.json

```json
[
  {
    "id": "sample_000001",
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": "The price rose from $21 to $30",
    "method": "numeric_metric_transform",
    "recognizer": "Regular",
    "success": true,
    "replacement": "$20 → $21",
    "failure_reason": null
  },
  {
    "id": "sample_000002",
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": null,
    "method": "numeric_metric_transform",
    "recognizer": "Regular",
    "success": false,
    "replacement": null,
    "failure_reason": "no_feature_found"
  }
]
```

`failure_reason` 取值：`no_feature_found` / `output_same_as_input` / `empty_output` / `llm_error`

### 4.3 method_stat.json

```json
{
  "method_name": "numeric_metric_transform",
  "recognizer_type": "Regular",
  "total_samples": 1000,
  "success_count": 847,
  "success_ratio": 0.847,
  "avg_feature_count": 1.3,
  "failure_reasons": {
    "no_feature_found": 120,
    "output_same_as_input": 33,
    "empty_output": 0,
    "llm_error": 0
  },
  "processing_time_sec": 4.2
}
```

### 4.4 construction_log.jsonl（逐条，机器可读）

每处理一条样本追加一行：

```json
{"ts": "2026-04-02T10:00:01", "sample_id": "sample_000001", "method": "numeric_metric_transform", "recognizer": "Regular", "input": "The price rose from $20 to $30", "features_found": {"numbers": ["$20", "$30"]}, "replacement": "$20 → $21", "output": "The price rose from $21 to $30", "success": true, "failure_reason": null, "time_ms": 3}
{"ts": "2026-04-02T10:00:01", "sample_id": "sample_000002", "method": "numeric_metric_transform", "recognizer": "Regular", "input": "He went to the park", "features_found": {"numbers": []}, "replacement": null, "output": null, "success": false, "failure_reason": "no_feature_found", "time_ms": 1}
```

### 4.5 construction_summary.txt（人类可读，run 结束后写入）

```
==============================
Method    : numeric_metric_transform
Recognizer: Regular
Date      : 2026-04-02 10:00:10
------------------------------
Total     : 1000
Success   : 847  (84.7%)
Failed    : 153  (15.3%)

Failure breakdown:
  no_feature_found    : 120 (78.4%)
  output_same_as_input:  33 (21.6%)
  empty_output        :   0  (0.0%)
  llm_error           :   0  (0.0%)

Avg feature count     : 1.30
Processing time       : 4.2s
==============================
```

### 4.6 dataset_methods_stat.json（全局，所有方法聚合）

```json
[
  {
    "id": "sample_000001",
    "pos": "The price rose from $20 to $30 in 5 days",
    "methods_feature_count": {
      "numeric_metric_transform":    {"Regular": 3, "LLM": 3},
      "entity_pronoun_substitution": {"Regular": 0, "LLM": 1},
      "direct_negation_attack":      {"Regular": 1, "LLM": 1},
      "double_negation_attack":      {"Regular": 0, "LLM": 0},
      "scope_degree_scaling":        {"Regular": 0, "LLM": 0},
      "logical_operator_rewrite":    {"Regular": 0, "LLM": 0},
      "role_swap":                   {"Regular": 0, "LLM": 0},
      "temporal_causal_inversion":   {"Regular": 0, "LLM": 0},
      "concept_hierarchy_shift":     {"Regular": 0, "LLM": 0},
      "premise_disruption":          {"Regular": 1, "LLM": 1}
    },
    "total_features_regular": 5,
    "total_features_llm": 6
  }
]
```

### 4.7 final_dataset.jsonl（所有方法 Regular 成功样本合并，去重）

```json
{"id": "sample_000001", "anchor": "...", "pos": "...", "neg": "...", "hard_neg": "...", "method": "numeric_metric_transform", "recognizer": "Regular"}
```

### 4.8 difference.md（每种方法一份）

```markdown
# {method_name} — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 84.7% | 79.3% |
| 平均特征数 | 1.30 | 1.52 |
| 处理时间 (s) | 4.2 | 38.6 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 120 | 87 |
| output_same_as_input | 33 | 29 |
| llm_error | 0 | 5 |

## 典型案例

### Regular 更优
- **输入**: ...
- **Regular**: 成功，替换 X → Y
- **LLM**: 失败，reason: no_feature_found
- **分析**: 正则对数值类句子更稳定

### LLM 更优
- **输入**: ...
- **Regular**: 失败，未识别到实体
- **LLM**: 成功，识别到 "Michael" 并替换
- **分析**: LLM 对命名实体识别更准确

## 结论

- Regular 推荐场景：含明确数值/逻辑词的句子
- LLM 推荐场景：含命名实体、复杂代词指代的句子
```

---

## 5. 模块接口

### 5.1 src_v2/data_loader.py

```python
@dataclass
class NLIRecord:
    id: str
    anchor: str
    pos: str
    neg: str

def load_parquet(path: str, sample_size: int = None, seed: int = 42) -> List[NLIRecord]
def save_preprocessed(records: List[NLIRecord], output_path: str) -> None
```

**复用**：`src.data_utils.normalize_text` 做字段清洗。

### 5.2 src_v2/feature_extractor.py

```python
def extract_regular(text: str) -> Dict[str, List[str]]
    # 调用 src.formatter._regex_extract + _safe_spacy_extract

def extract_llm(text: str, llm_engine: LocalLLMEngine) -> Dict[str, List[str]]
    # 调用 src.formatter._llm_extract

def count_method_features(features: Dict, method_name: str) -> int
    # 返回该方法可用的特征数量（用于 dataset_methods_stat）
```

`count_method_features` 逻辑：

| 方法 | 计数依据 |
|---|---|
| numeric_metric_transform | `len(features["numbers"])` |
| entity_pronoun_substitution | `len(features["entities"]) + len(features["pronouns"])` |
| scope_degree_scaling | `len(features["degree_words"])` |
| direct_negation_attack | `1` if no negation present else `0` |
| double_negation_attack | `len(features["negations"])` |
| logical_operator_rewrite | `len(features["logic_words"])` |
| role_swap | `min(len(subjects), len(objects))` |
| temporal_causal_inversion | `len(features["sequence_words"])` |
| concept_hierarchy_shift | `len(features["entities"])` |
| premise_disruption | `1` always (fallback guaranteed) |

### 5.3 src_v2/builder.py

```python
@dataclass
class RunResult:
    method_name: str
    recognizer_type: str
    records: List[Dict]      # constructed_data 内容
    stats: Dict              # method_stat 内容
    feature_counts: Dict     # {sample_id: feature_count}，供 analyzer 汇总

class PipelineRunner:
    def __init__(self, records: List[NLIRecord], method_name: str,
                 recognizer_type: str, output_dir: Path,
                 llm_engine: LocalLLMEngine = None)

    def run(self) -> RunResult
        # 流程：
        # for each record:
        #   1. extract_regular / extract_llm → features
        #   2. count_method_features → feature_count
        #   3. apply_method(method_name, text=record.pos, features)
        #   4. ensure_text3_valid 验证
        #   5. 追加写 construction_log.jsonl
        # 结束后写 construction_summary.txt、constructed_data.json、method_stat.json
```

**复用**：
- `src.constructors.apply_method`（全部 10 个方法）
- `src.main_generator.ensure_text3_valid`（输出验证）
- `src.llm_engine.LocalLLMEngine`（LLM 推理）

### 5.4 src_v2/analyzer.py

```python
def build_dataset_methods_stat(
    all_results: Dict[str, Dict[str, RunResult]]  # {method: {recognizer: RunResult}}
) -> List[Dict]
    # 聚合所有方法的 feature_counts → dataset_methods_stat.json

def aggregate_final_dataset(
    all_results: Dict[str, Dict[str, RunResult]]
) -> List[Dict]
    # 合并所有方法 Regular 成功样本，按 sample_id 去重（保留第一个）→ final_dataset.jsonl

def generate_difference_report(
    method_name: str,
    regular_result: RunResult,
    llm_result: RunResult
) -> str
    # 生成 difference.md 字符串
```

### 5.5 scripts_v2/run_stage2.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_path` | 必填 | parquet 文件路径 |
| `--output_base` | `Data/` | 输出根目录 |
| `--sample_size` | `1000` | 随机采样数，None 为全量 |
| `--methods` | `all` | 逗号分隔方法名，或 `all` |
| `--recognizer` | `both` | `regular` / `llm` / `both` |
| `--llm_model` | `""` | LLM 模型路径（recognizer=llm/both 时必填） |
| `--seed` | `42` | 随机种子 |

执行顺序：
1. 加载并采样数据 → `preprocessed_data.json`
2. 对每个 method × recognizer 组合运行 `PipelineRunner`
3. 每个方法两条路径都完成后，运行 `generate_difference_report` → `difference.md`
4. 全部方法完成后，运行 `build_dataset_methods_stat` → `dataset_methods_stat.json`
5. 运行 `aggregate_final_dataset` → `final_dataset.jsonl`

---

## 6. 复用边界

| 现有模块 | Stage 2 用法 | 修改 |
|---|---|---|
| `src/constructors.py` | `apply_method` 全部 10 个方法 | 无 |
| `src/llm_engine.py` | `LocalLLMEngine` | 无 |
| `src/formatter._regex_extract` | `extract_regular` 内部调用 | 无 |
| `src/formatter._safe_spacy_extract` | `extract_regular` 内部调用 | 无 |
| `src/formatter._llm_extract` | `extract_llm` 内部调用 | 无 |
| `src/data_utils.normalize_text` | `data_loader` 字段清洗 | 无 |
| `src/main_generator.ensure_text3_valid` | `builder` 输出验证 | 无 |

`src/` 目录零修改，Stage 1 pipeline 完全不受影响。

---

## 7. 新增依赖

```
pandas          # parquet 读取（pyarrow backend）
pyarrow         # pandas parquet 依赖
```

其余依赖均已在 `requirements.txt` 中。

---

## 8. 测试策略

- 用 1000 条样本跑通全流程，验证目录结构和所有输出文件格式
- 对每个方法检查 `method_stat.json` 中 `success_count + failure_counts == total_samples`
- 检查 `dataset_methods_stat.json` 中所有 sample_id 覆盖完整
- LLM 路径在 `--recognizer regular` 时完全跳过，不影响 Regular 路径结果
