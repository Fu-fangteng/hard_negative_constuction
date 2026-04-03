# 二阶段项目实现 Instruction：规范化困难负样本训练集构造 Pipeline

## 一、项目概述

### 项目目标
以 nli_for_simcse 的三元组数据（{anchor, positive, negative}）为输入，通过自动化构造困难负样本，生成四元组数据 {anchor, positive, negative, hard_negative}，为对比学习等任务提供更具挑战性的训练集。

### 项目创新点
- **双识别方法对比**：同时支持正则表达式（Regular）和 LLM-based（如 Qwen）两种实体识别方式
- **构造方法模块化**：每个构造方法独立模块化实现，支持灵活组合
- **完整统计与监控**：详细记录识别、替换、失败情况，便于方法对比与分析

---

## 二、项目架构

### 2.1 整体流程图

```
原始数据 (nli_train.jsonl)
    ↓
数据预处理 (preprocessing)
    ↓
数据采样 (sampling) [可选]
    ↓
特征识别与格式化 (feature_extraction)
    ├─ Regular 方式
    └─ LLM 方式
    ↓
困难负样本构造 (construction)
    ├─ 构造方法 1: numeric_metric_transform
    │   ├─ LLM 版本
    │   └─ Regular 版本
    ├─ 构造方法 2: entity_pronoun_substitution
    │   ├─ LLM 版本
    │   └─ Regular 版本
    └─ 构造方法 N: ...
    ↓
结果聚合与分析 (aggregation & analysis)
    ├─ 构造统计 (method_stat.json)
    ├─ 构造日志 (construction_log)
    └─ 方法对比 (difference.md)
```

### 2.2 代码模块组织

```
hard_neg/
├── Data/
│   ├── original_data/           # 原始数据
│   ├── preprocessed_data/       # 预处理后数据
│   └── processed_data/          # 处理后数据（按方法组织）
├── src_v2/                      # 第二阶段核心代码
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载与预处理
│   ├── feature_extractor.py     # 特征识别（Regular + LLM）
│   ├── constructors/            # 构造方法库
│   │   ├── __init__.py
│   │   ├── base.py              # 基类定义
│   │   ├── numeric_metric.py    # 数值变换
│   │   ├── entity_pronoun.py    # 实体代词替换
│   │   ├── scope_degree.py      # 范围程度缩放
│   │   ├── negation.py          # 否定攻击
│   │   ├── logical_operator.py  # 逻辑算子
│   │   ├── role_swap.py         # 角色互换
│   │   ├── temporal_causal.py   # 时序因果倒置
│   │   ├── concept_hierarchy.py # 概念层级
│   │   └── premise_disruption.py # 前提破坏
│   ├── builder.py               # 构造流程调度
│   └── analyzer.py              # 结果分析与统计
├── scripts_v2/                  # 第二阶段脚本
│   ├── preprocess.py            # 预处理脚本
│   ├── construct_dataset.py     # 主构造脚本
│   └── analyze_results.py       # 结果分析脚本
├── configs/                     # 配置文件
│   └── construction_config.yaml # 构造方法配置
├── STAGE2_INSTRUCTION.md        # 本文件
└── README_STAGE2.md             # 阶段二总结
```

---

## 三、数据格式规范

### 3.1 原始数据格式 (nli_train.jsonl)

```json
{"id": 1, "anchor": "...", "positive": "...", "negative": "..."}
{"id": 2, "anchor": "...", "positive": "...", "negative": "..."}
```

### 3.2 预处理后数据格式 (preprocessed_data.json)

```json
[
  {"id": 1, "anchor": "...", "pos": "...", "neg": "..."},
  {"id": 2, "anchor": "...", "pos": "...", "neg": "..."}
]
```

注：统一使用简短字段名 (pos, neg)；字段已清洗空白。

### 3.3 特征提取结果 (processed_data/{method}/{recognizer}/formatted_data.json)

```json
[
  {
    "id": 1,
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "features": {
      "entities": ["entity1", "entity2"],
      "numbers": ["123", "45.6"],
      "logic_marker": "some",
      "negation": false,
      "...": "..."
    },
    "recognizer_type": "Regular | LLM"
  }
]
```

### 3.4 构造后数据格式 (processed_data/{method}/{recognizer}/constructed_data.json)

```json
[
  {
    "id": 1,
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": "...",
    "method": "numeric_metric_transform",
    "recognizer": "Regular",
    "success": true,
    "replaced_element": ["123 → 456"],
    "log": "Replaced numeric value 123 with 456"
  },
  {
    "id": 2,
    "anchor": "...",
    "pos": "...",
    "neg": "...",
    "hard_neg": null,
    "method": "entity_pronoun_substitution",
    "recognizer": "LLM",
    "success": false,
    "replaced_element": [],
    "log": "No entity extracted by LLM"
  }
]
```

### 3.5 统计文件格式 (processed_data/{method}/{recognizer}/method_stat.json)

```json
{
  "method_name": "numeric_metric_transform",
  "recognizer_type": "Regular",
  "total_samples": 100,
  "success_count": 85,
  "success_ratio": 0.85,
  "avg_replacement_count": 1.2,
  "feature_extraction_success": 0.92,
  "failure_reasons": {
    "no_feature_found": 10,
    "replacement_failed": 4,
    "output_same_as_input": 1
  },
  "processing_time_sec": 12.34
}
```

### 3.6 构造日志格式 (processed_data/{method}/{recognizer}/construction_log)

```
[2025-03-31 10:00:00] INFO: Processing method: numeric_metric_transform, recognizer: Regular
[2025-03-31 10:00:00] INFO: Loading preprocessed data from preprocessed_data.json
[2025-03-31 10:00:01] INFO: Total samples to process: 100
[2025-03-31 10:00:02] INFO: [Sample 1] Feature extraction: 2 numbers found
[2025-03-31 10:00:02] INFO: [Sample 1] Replacement: 123 → 456 successful
[2025-03-31 10:00:02] INFO: [Sample 1] Final hard_neg generated successfully
[2025-03-31 10:00:03] WARN: [Sample 2] No feature found, skipping
[2025-03-31 10:00:04] ERROR: [Sample 3] Replacement failed due to regex error
[2025-03-31 10:00:10] INFO: Processing complete. Success: 85/100, Time: 8.5s
```

### 3.7 差异对比文件格式 (processed_data/{method}/difference.md)

```markdown
# {method_name} 两种识别方法对比

## 总体对比

| 指标 | Regular | LLM |
|-----|---------|-----|
| 成功率 | XX.X% | YY.Y% |
| 平均识别个数 | A | B |
| 平均替换个数 | C | D |
| 处理时间 (秒) | E | F |

## 典型案例分析

### 案例1：Regular 更优
- **输入**: xxx
- **Regular**: 成功识别X个特征，替换Y个，得到: xxx
- **LLM**: 识别失败，reason: xxx
- **分析**: Regular 的规则更可靠

### 案例2：LLM 更优
- **输入**: xxx
- **Regular**: 误识别/无法识别
- **LLM**: 正确识别，得到: xxx
- **分析**: LLM 的语义理解更强

## 结论

推荐场景：
- Regular: 针对数值变换、明确结构的句子
- LLM: 针对复杂实体替换、语义理解需求
```

---

## 四、核心模块设计

### 4.1 数据加载与预处理 (src_v2/data_loader.py)

#### 接口定义

```python
def load_nli_data(input_path: str) -> List[Dict]:
    """
    加载原始 nli_for_simcse 数据。
    
    Args:
        input_path: jsonl 文件路径
    
    Returns:
        List[Dict]: 标准化数据 [{"id", "anchor", "pos", "neg"}, ...]
    """
    pass

def preprocess_data(records: List[Dict], 
                    remove_duplicates: bool = True,
                    clean_whitespace: bool = True) -> List[Dict]:
    """
    数据清洗与标准化。
    
    Args:
        records: 原始数据
        remove_duplicates: 是否删除重复
        clean_whitespace: 是否清理空白
    
    Returns:
        List[Dict]: 预处理后数据
    """
    pass

def sample_data(records: List[Dict], 
                sample_size: int = None, 
                seed: int = 42) -> List[Dict]:
    """
    数据采样。
    
    Args:
        records: 原始数据
        sample_size: 采样数量，None 表示全部
        seed: 随机种子
    
    Returns:
        List[Dict]: 采样后数据
    """
    pass

def save_data(records: List[Dict], output_path: str):
    """保存预处理数据到 JSON 文件"""
    pass
```

### 4.2 特征识别 (src_v2/feature_extractor.py)

#### 接口定义

```python
class FeatureExtractor(ABC):
    """特征识别的抽象基类"""
    
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取特征。
        
        Returns:
            {
                "entities": [...],
                "numbers": [...],
                "logic_markers": [...],
                "negation": bool,
                ...
            }
        """
        pass

class RegularFeatureExtractor(FeatureExtractor):
    """基于正则表达式的特征识别"""
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        识别：
        - 数值（整数、浮点数、百分比、货币）
        - 命名实体（简单启发式）
        - 逻辑词汇（all/some/most/never/maybe）
        - 否定词标记
        """
        pass

class LLMFeatureExtractor(FeatureExtractor):
    """基于 LLM 的特征识别（如 Qwen）"""
    
    def __init__(self, model_name: str = "qwen-7b", device: str = "cuda"):
        """初始化 LLM 模型"""
        pass
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        调用 LLM (Qwen) 进行结构化特征抽取。
        
        提示词：提取实体、数值、逻辑词汇等，返回 JSON 格式
        """
        pass

def format_dataset(records: List[Dict], 
                   extractor_type: str = "regular",  # "regular" or "llm"
                   **kwargs) -> List[Dict]:
    """
    为整个数据集抽取特征。
    
    Args:
        records: 预处理数据
        extractor_type: "regular" 或 "llm"
        **kwargs: 传递给 Extractor 的参数
    
    Returns:
        List[Dict]: 包含特征的数据
    """
    pass
```

### 4.3 构造方法基类 (src_v2/constructors/base.py)

#### 接口定义

```python
class HardNegativeConstructor(ABC):
    """困难负样本构造方法的抽象基类"""
    
    def __init__(self, recognizer_type: str = "regular"):
        """
        Args:
            recognizer_type: "regular" 或 "llm"
        """
        self.recognizer_type = recognizer_type
    
    @abstractmethod
    def construct(self, text: str, features: Dict[str, Any]) -> Optional[str]:
        """
        构造困难负样本。
        
        Args:
            text: 输入文本（positive 句子）
            features: 特征字典（从 FeatureExtractor 得到）
        
        Returns:
            str: 困难负样本文本，失败返回 None
        """
        pass
    
    def _validate_output(self, original: str, generated: Optional[str]) -> bool:
        """
        验证生成的文本：
        1. 不为空
        2. 与原文本不完全相同
        """
        pass
```

### 4.4 具体构造方法示例 (src_v2/constructors/)

#### 例1：数值变换 (numeric_metric.py)

```python
class NumericMetricTransformer(HardNegativeConstructor):
    """
    改变数字、百分比、货币单位等。
    支持 Regular 和 LLM 两种方式。
    """
    
    def __init__(self, recognizer_type: str = "regular"):
        super().__init__(recognizer_type)
        self.method_name = "numeric_metric_transform"
    
    def construct(self, text: str, features: Dict[str, Any]) -> Optional[str]:
        """
        Regular: 正则找数字，随机加减
        LLM: 提示 LLM 改变数值但保持句法
        """
        if self.recognizer_type == "regular":
            return self._construct_regular(text, features)
        else:
            return self._construct_llm(text, features)
    
    def _construct_regular(self, text: str, features: Dict) -> Optional[str]:
        """基于正则表达式的实现"""
        numbers = features.get("numbers", [])
        if not numbers:
            return None
        
        result = text
        for num in numbers:
            # 随机改变数值
            modified_num = str(float(num) * random.uniform(0.5, 2.0))
            result = result.replace(num, modified_num, 1)
        
        return result if result != text else None
    
    def _construct_llm(self, text: str, features: Dict) -> Optional[str]:
        """基于 LLM 的实现"""
        prompt = f"""
        修改以下句子中的数值，使其改变但保持句法结构和逻辑。
        句子: {text}
        
        要求：
        1. 只修改数值部分
        2. 句法结构保持不变
        3. 返回修改后的句子
        
        修改后的句子:
        """
        # 调用 LLM，返回结果
        pass
```

#### 例2：实体代词替换 (entity_pronoun.py)

```python
class EntityPronounSubstitution(HardNegativeConstructor):
    """
    替换人名、地名、组织名等实体，或修改指代。
    """
    
    def construct(self, text: str, features: Dict[str, Any]) -> Optional[str]:
        if self.recognizer_type == "regular":
            return self._construct_regular(text, features)
        else:
            return self._construct_llm(text, features)
    
    def _construct_regular(self, text: str, features: Dict) -> Optional[str]:
        """正则启发式识别实体并替换"""
        entities = features.get("entities", [])
        if not entities:
            return None
        
        # 从预定义的替换字典中选择替换实体
        result = text
        for entity in entities:
            replacement = self._get_replacement(entity)
            if replacement:
                result = result.replace(entity, replacement, 1)
        
        return result if result != text else None
    
    def _construct_llm(self, text: str, features: Dict) -> Optional[str]:
        """用 LLM 识别和替换实体"""
        prompt = f"""
        在以下句子中识别实体（人名、地名、组织名等），
        并将其替换为合理的其他实体，保持句子逻辑通顺。
        
        句子: {text}
        
        要求：
        1. 识别所有主要实体
        2. 用合理的替代实体替换
        3. 保持句子语法和逻辑
        
        替换后的句子:
        """
        pass
```

### 4.5 构造流程调度 (src_v2/builder.py)

#### 接口定义

```python
class HardNegativeBuilder:
    """主流程调度器"""
    
    def __init__(self, 
                 formatted_data: List[Dict],
                 method_name: str,
                 recognizer_type: str = "regular"):
        """
        Args:
            formatted_data: 包含特征的数据
            method_name: 构造方法名，如 "numeric_metric_transform"
            recognizer_type: "regular" 或 "llm"
        """
        self.formatted_data = formatted_data
        self.method_name = method_name
        self.recognizer_type = recognizer_type
        self.constructor = self._get_constructor()
        self.results = []
        self.stats = {}
    
    def _get_constructor(self) -> HardNegativeConstructor:
        """根据 method_name 获取对应的构造器"""
        constructors = {
            "numeric_metric_transform": NumericMetricTransformer,
            "entity_pronoun_substitution": EntityPronounSubstitution,
            # ... 其他方法
        }
        return constructors[self.method_name](self.recognizer_type)
    
    def build(self) -> List[Dict]:
        """
        构造所有样本的困难负样本。
        
        Returns:
            List[Dict]: 构造结果，包含 hard_neg, success, log 等字段
        """
        for item in self.formatted_data:
            hard_neg = self.constructor.construct(
                text=item["pos"],
                features=item.get("features", {})
            )
            
            result = {
                **item,
                "hard_neg": hard_neg,
                "method": self.method_name,
                "recognizer": self.recognizer_type,
                "success": hard_neg is not None,
                "log": "Construction successful" if hard_neg else "Construction failed"
            }
            self.results.append(result)
        
        return self.results
    
    def compute_stats(self) -> Dict:
        """计算统计指标"""
        success_count = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        
        self.stats = {
            "method_name": self.method_name,
            "recognizer_type": self.recognizer_type,
            "total_samples": total,
            "success_count": success_count,
            "success_ratio": success_count / total if total > 0 else 0,
            # ... 其他统计
        }
        return self.stats
    
    def save_results(self, output_dir: str):
        """保存结果、统计、日志"""
        pass
```

### 4.6 结果分析 (src_v2/analyzer.py)

#### 接口定义

```python
class ResultAnalyzer:
    """结果分析与对比"""
    
    def __init__(self, method_name: str, output_base_dir: str):
        """
        Args:
            method_name: 方法名
            output_base_dir: 方法的输出目录
        """
        self.method_name = method_name
        self.output_base_dir = output_base_dir
    
    def compare_recognizers(self) -> Dict:
        """
        对比 Regular 和 LLM 两种识别方式的效果。
        
        Returns:
            {
                "success_ratio_regular": x,
                "success_ratio_llm": y,
                "avg_replacement_regular": a,
                "avg_replacement_llm": b,
                "typical_cases": [...]
            }
        """
        pass
    
    def generate_difference_report(self) -> str:
        """
        生成 difference.md 对比报告。
        """
        pass
    
    def sample_case_study(self, num_cases: int = 3) -> List[Dict]:
        """
        抽样案例分析，突出两种方法的优劣。
        """
        pass
```

---

## 五、执行流程

### 5.1 完整执行步骤

#### Step 1: 预处理
```bash
python scripts_v2/preprocess.py \
  --input_path Data/original_data/nli_train.jsonl \
  --output_path Data/preprocessed_data/preprocessed_data.json \
  --sample_size 1000 \
  --seed 42
```

#### Step 2: 特征抽取
```bash
python scripts_v2/construct_dataset.py \
  --input_path Data/preprocessed_data/preprocessed_data.json \
  --output_base Data/processed_data \
  --method numeric_metric_transform \
  --recognizer regular \
  --save_features
```

#### Step 3: 困难负样本构造
```bash
python scripts_v2/construct_dataset.py \
  --input_path Data/preprocessed_data/preprocessed_data.json \
  --output_base Data/processed_data \
  --method numeric_metric_transform \
  --recognizer regular
```

#### Step 4: 对比分析
```bash
python scripts_v2/analyze_results.py \
  --method_name numeric_metric_transform \
  --output_base Data/processed_data
```

### 5.2 主脚本参数表

#### construct_dataset.py

| 参数 | 类型 | 默认 | 说明 |
|-----|------|------|------|
| --input_path | str | 必需 | 预处理数据路径 |
| --output_base | str | Data/processed_data | 输出基础目录 |
| --method | str | 必需 | 构造方法 (numeric_metric_transform 等) |
| --recognizer | str | regular | 识别方式 (regular 或 llm) |
| --llm_model | str | qwen-7b | LLM 模型名 (--recognizer=llm 时) |
| --device | str | cuda | 计算设备 |
| --batch_size | int | 32 | 批处理大小 |
| --save_features | bool | False | 是否保存特征提取结果 |
| --save_log | bool | True | 是否保存构造日志 |
| --seed | int | 42 | 随机种子 |

---

## 六、实现约定与规范

### 6.1 命名规范

- 构造方法类命名：`<MethodNameInPascalCase>` (e.g., NumericMetricTransformer)
- 文件名：`<method_name_in_snake_case>.py` (e.g., numeric_metric.py)
- 方法内部函数：`_<function_name>` (私有)，`<function_name>` (公开)
- 变量名：`snake_case`

### 6.2 代码质量

- 所有方法需包含详细的 docstring（参数、返回值、异常）
- 使用类型提示 (Python typing)
- 错误处理：使用 try-except 和自定义异常
- 日志：使用 logging 模块，支持多级别日志

### 6.3 配置管理

使用 YAML 配置文件 `configs/construction_config.yaml`：

```yaml
methods:
  numeric_metric_transform:
    enabled: true
    description: "数值与度量变换"
    recognizers: ["regular", "llm"]
    
  entity_pronoun_substitution:
    enabled: true
    description: "实体/指代置换"
    recognizers: ["regular", "llm"]

recognizers:
  regular:
    type: "rule-based"
  llm:
    type: "model-based"
    model_name: "qwen-7b"
    device: "cuda"
    parameters:
      temperature: 0.2
      max_tokens: 256
```

### 6.4 错误处理

定义自定义异常：

```python
class ConstructionError(Exception):
    """构造失败"""
    pass

class FeatureExtractionError(Exception):
    """特征抽取失败"""
    pass

class LLMInferenceError(Exception):
    """LLM 推理失败"""
    pass
```

---

## 七、数据质量保证

### 7.1 验证规则

每个构造方法需实现以下验证：

1. **非空检查**：hard_neg 不为空
2. **差异检查**：hard_neg ≠ pos（规范化后）
3. **长度检查**：hard_neg 长度在合理范围 (0.5x ~ 2.0x pos 长度)
4. **语言检查**（可选）：使用 fastText 验证语言一致性

```python
def validate_hard_negative(original: str, generated: str) -> Tuple[bool, str]:
    """
    验证生成的困难负样本。
    
    Returns:
        (success: bool, reason: str)
    """
    # 实现上述规则
    pass
```

### 7.2 统计监控

每个输出文件需包含：
- 成功率（success_ratio）
- 平均替换个数（avg_replacement_count）
- 失败原因统计（failure_reasons）
- 处理时间

---

## 八、示例与案例

### 案例1：数值变换 (numeric_metric_transform)

**输入**：
```
"The temperature increased from 20C to 30C in 5 minutes"
```

**特征提取（Regular）**：
```python
{
  "numbers": ["20", "30", "5"],
  "units": ["C", "minutes"],
  ...
}
```

**构造结果**：
```
"The temperature increased from 15C to 35C in 7 minutes"
```

**特征提取（LLM）**：
```python
{
  "numeric_entities": [
    {"value": "20", "unit": "C", "role": "initial_temp"},
    {"value": "30", "unit": "C", "role": "final_temp"},
    {"value": "5", "unit": "minutes", "role": "duration"}
  ],
  ...
}
```

**构造结果**：
```
"The temperature increased from 15C to 35C in 7 minutes"
```

### 案例2：实体替换 (entity_pronoun_substitution)

**输入**：
```
"Michael went to Paris and met with the mayor"
```

**Regular 识别**：
```python
{
  "entities": ["Michael", "Paris"]
}
```

**结果**：
```
"John went to London and met with the mayor"
```

**LLM 识别**：
```python
{
  "entities": [
    {"text": "Michael", "type": "PERSON"},
    {"text": "Paris", "type": "GPE"}
  ]
}
```

**结果**：
```
"Sarah went to Tokyo and met with the mayor"
```

---

## 九、测试与验证

### 9.1 单元测试

每个构造方法需包含单元测试：

```
tests/
├── test_numeric_metric.py
├── test_entity_pronoun.py
├── test_feature_extraction.py
└── ...
```

### 9.2 集成测试

```bash
pytest tests/ -v --cov=src_v2
```

### 9.3 手动验证

随机抽样 50 条生成的 hard_neg，人工评估：
- 逻辑合理性
- 困难程度
- 与原句的语义差异程度

---

## 十、交付清单

完成第二阶段后，应包含：

- [ ] 完整的源代码 (src_v2/)
- [ ] 执行脚本 (scripts_v2/)
- [ ] 配置文件 (configs/)
- [ ] 测试套件 (tests/)
- [ ] 生成的数据集 (Data/processed_data/)
- [ ] 方法对比报告 (difference.md)
- [ ] 项目文档与使用指南 (README_STAGE2.md)
- [ ] 统计指标汇总表

---

## 十一、后续扩展方向

1. **更多构造方法**：补充其他 9 种方法
2. **更多 LLM 模型**：支持 GPT-4、Claude 等
3. **自动方法选择**：根据特征自动分配最优方法
4. **迭代改进**：基于困难度反馈优化参数
5. **多语言支持**：扩展至其他语言数据

---

## 十二、参考与依赖

### 核心依赖
- numpy, pandas
- transformers (for LLM loading)
- torch/tensorflow
- pyyaml (for config)
- tqdm (for progress bar)

### 可选依赖
- spacy (for NER)
- fasttext (for language detection)
- matplotlib (for visualization)

### 相关文档
- [hard_neg_construction_method.md](./hard_neg_construction_method.md)：详细的构造方法说明
- [README.md](./README.md)：项目总体说明
- [stage2.md](./stage2.md)：阶段二需求
