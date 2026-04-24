# 改进建议：Hard Negative 构造与训练

> 基于现有实验结果（STSBenchmark: Base=0.8203, NLI=**0.8392**, Regular=0.8358, LLM=0.8317, Combined=0.8311）

---

## 问题一：构造数据质量

### 1a. Regular 路径的语法错误

#### 根因分析

`direct_negation_attack` 在处理含缩写的否定句时，依赖以下正则将 `"doesn't"` 还原为 `"does"`：

```python
re.sub(r"\b(\w+?)n['']?t\b", r"\1", text, count=1)
```

正则中 `(\w+?)` 是**懒惰匹配**，会尽量少匹配字符。这对大多数缩写正确，但对以下特殊形式产生语法错误：

| 原始缩写 | 当前输出 | 正确输出 |
|---------|---------|---------|
| `can't` | `ca` | `can` |
| `won't` | `wo` | `will`（或至少 `won`） |
| `shan't` | `sha` | `shall` |

错误示例：`"He can't swim"` → `"He ca swim"` ——这样的样本进入训练集会污染模型。

第二个问题出在句首 `"no"` 的删除逻辑：

```python
pat = re.compile(rf"\bno\b\s*", flags=re.IGNORECASE)
```

当句子以 `"No"` 开头时，删除后残句无法独立成句：
- `"No one was hurt."` → `"one was hurt."` ✗
- `"No shooting occurred."` → `"shooting occurred."` ✓（可接受）

第三个问题是 `_negate_without_aux` 的正则 fallback：识别动词的启发式依赖 `-ed`/`-s` 词尾，会把结尾为 `-ness`、`-tion`、`-less` 的名词误判为动词（尽管已有 `_NON_VERB_ENDS` 黑名单，但覆盖不全）。

#### 改进方案

**修复 1：缩写还原用查找表替代正则**

```python
_CONTRACTION_MAP = {
    "can't": "can",   "won't": "will",  "shan't": "shall",
    "don't": "do",    "doesn't": "does", "didn't": "did",
    "isn't": "is",    "aren't": "are",   "wasn't": "was",
    "weren't": "were", "hasn't": "has",  "haven't": "have",
    "hadn't": "had",  "shouldn't": "should", "wouldn't": "would",
    "couldn't": "could", "mightn't": "might", "mustn't": "must",
}
```

对输入先做缩写查表替换，失败才回退到正则。

**修复 2：`"no"` 删除时检查残句合法性**

删除 `"no"` 后检查剩余句子首个词是否为代词（`one/body/thing`），是则跳过此操作改用其他策略。

**修复 3：后处理过滤器**

在 `PipelineRunner.run()` 中，对 Regular 路径的输出增加一个后处理过滤层，过滤以下情况：
- 输出长度与原文相差超过 40%（过度截断或冗余扩展）
- 输出以小写字母开头（删除句首词导致的残句）
- 输出以孤立功能词（`"a"/"an"/"the"/"of"` 等）开头

---

### 1b. LLM 路径的语义伪正样本

#### 根因分析

`_parse_llm_output` 仅做**表层文本过滤**（空输出、与原文相同、末尾孤立 `"not"`），无法捕获以下情况：

**问题 1：`premise_disruption` 只加前缀，语义完全不变**

```
pos     : "Vaccines are effective."
hard_neg: "Contrary to what was stated, vaccines are effective."
```

两者核心语义命题完全相同，嵌入相似度接近 1.0，根本不是"困难"负样本。

**问题 2：LLM 生成时改写了 positive 而不是制造矛盾**

小模型（1.7B）在指令遵循上不稳定，有时会产生语义等价的改写（同义词替换、语序调整），导致 hard_neg ≈ positive。

**问题 3：模型自身幻觉**

LLM 有时生成与 positive 语义正相关但表面不同的句子，例如：
```
pos     : "The patient recovered quickly."
hard_neg: "The patient healed rapidly."   ← 语义正样本
```

#### 改进方案

**改进 1：嵌入相似度过滤（核心修复）**

在 `PipelineRunner.run()` 中，对 LLM 生成的 hard_neg 增加语义验证步骤：

```python
# 使用与训练相同的 base model 做验证
from sentence_transformers import SentenceTransformer
_validator = SentenceTransformer("all-MiniLM-L6-v2")

def _is_semantic_positive(pos: str, hard_neg: str, threshold: float = 0.90) -> bool:
    embs = _validator.encode([pos, hard_neg], normalize_embeddings=True)
    return float(embs[0] @ embs[1]) >= threshold
```

如果 `sim(pos, hard_neg) ≥ 0.90`，则标记为 `failure_reason="semantic_positive"` 并丢弃。阈值 0.90 可通过检查少量样本校准。

**改进 2：`premise_disruption` 从 LLM 路径中排除**

该方法本质上只是在 pos 前追加一个矛盾前缀，无论规则还是 LLM 都无法真正改变语义。应在 LLM 路径中对此方法直接返回 `None`（等效于不支持此方法的 LLM 构造）。

**改进 3：加强 LLM Prompt 中的排斥指令**

在 `CONSTRUCTION_SYSTEM_PROMPT` 中增加明确限制：

```
The output must describe a DIFFERENT or OPPOSITE situation.
Do NOT simply rephrase, paraphrase, or reorder words.
Do NOT use synonyms that preserve the same meaning.
```

---

## 问题二：训练损失函数

### 现状分析

当前代码使用 `MultipleNegativesRankingLoss`（MNRL），数据列为 `(anchor, positive, negative)`。

MNRL 的优化目标是 InfoNCE 风格的对比学习：对每个 anchor，在 batch 内把对应 positive 排到最高，其余 batch item（含显式 hard_neg）都作为负样本。

**问题**：MNRL 将 hard_neg 与随机 batch 负样本**平等对待**，没有显式惩罚"边界模糊"的困难对。当 `sim(anchor, hard_neg)` 本身很高时，模型对它的梯度贡献已经很大，但 loss 形式不会**明确要求**拉开 positive 和 hard_neg 之间的间距。

此外，batch size=128 时，每个 anchor 看到 127 个其他 positive（+ 1 个显式 hard_neg）作为负样本——hard_neg 的信号被大量随机负样本稀释。

### 改进方案

#### 方案 A：加入显式 `TripletLoss`（最直接）

`TripletLoss` 直接优化 `max(0, sim(a, neg) - sim(a, pos) + margin)`，迫使模型拉开正负样本的间距边界。适合在 MNRL 之后做 fine-tuning（curriculum 第二阶段）：

```python
from sentence_transformers.losses import TripletLoss, SiameseDistanceMetric

triplet_loss = TripletLoss(
    model=model,
    distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
    triplet_margin=0.2,   # 要求 cos(a,p) - cos(a,n) ≥ 0.2
)
```

数据列需调整为 `(anchor, positive, negative)`（与当前格式一致）。

#### 方案 B：调高 MNRL 温度参数（低成本改进）

MNRL 的 `scale` 参数控制 logit 温度（默认 20）。提高 scale 会使 softmax 更"尖锐"，给困难对提供更大梯度：

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss
loss = MultipleNegativesRankingLoss(model, scale=30.0)
```

注意：scale 过大会导致训练不稳定，建议范围 20–40。

#### 方案 C：Curriculum Learning（推荐）

分两阶段训练，逐步提高任务难度：

**Phase 1**（第 1 epoch，全量 NLI 数据 + 原始 neg）：用 MNRL 学习通用语义对齐：
```python
# 数据：(anchor, positive)，negative 来自 NLI 原始负样本
loss_phase1 = MultipleNegativesRankingLoss(model, scale=20.0)
```

**Phase 2**（第 2 epoch，仅 hard neg 数据）：用 TripletLoss 精调边界：
```python
# 数据：(anchor, positive, hard_neg)
loss_phase2 = TripletLoss(model, triplet_margin=0.15)
```

Phase 2 的学习率应比 Phase 1 降低 5–10 倍（如 `2e-6`）。

#### 方案 D：`GISTEmbedLoss`（引导批内难例挖掘）

Sentence Transformers 提供 `GISTEmbedLoss`，使用一个 guide 模型从 batch 内选择更难的负样本，叠加在显式 hard_neg 之上：

```python
from sentence_transformers.losses import GISTEmbedLoss
from sentence_transformers import SentenceTransformer

guide_model = SentenceTransformer("all-MiniLM-L6-v2")  # 或更大的 teacher model
loss = GISTEmbedLoss(model, guide=guide_model)
```

此方法的前提是 guide model 质量较高，适合有充足算力时使用。

---

## 问题三：MTEB 榜单表现

### 3a. 确认 MTEB 官方接口

`evaluate/run_eval.py` 已使用 **官方 `mteb` Python 包**，调用方式正确：

```python
import mteb
task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
benchmark = mteb.MTEB(tasks=task_objects)
results = benchmark.run(st_model, output_folder=str(out_dir))
```

提取的指标为 `spearman_cosine`，与 MTEB 榜单的主指标一致。

**当前已知问题**：现有评估只跑了 `STSBenchmark` 一个任务（因为上次在 HPC 上可能只运行了 debug 模式或受时间限制），但 MTEB STS 榜单基于 7 个任务的平均值：`STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R`。

**建议**：完整运行 7 个任务后，再与 MTEB 榜单上 `all-MiniLM-L6-v2` 的官方分数对齐，确认本地复现数值无误。`all-MiniLM-L6-v2` 在榜单上的 STS 均值约为 **0.6823**（注意：榜单部分任务包括 STS16 等，而非仅 STSBenchmark），本地跑 STSBenchmark 单任务的 0.8203 与这并不矛盾。

### 3b. 提高 MTEB 表现

#### 现状差距

| 模型 | STSBenchmark | Δ vs NLI Baseline |
|------|-------------|------------------|
| Base（all-MiniLM-L6-v2）| 0.8203 | −0.0189 |
| NLI Baseline | **0.8392** | — |
| Regular Hard Neg | 0.8358 | −0.0034 |
| LLM Hard Neg | 0.8317 | −0.0075 |
| Combined Hard Neg | 0.8311 | −0.0081 |

**核心问题**：Hard Neg 训练提升了 STSBenchmark（+0.0108~+0.0155），但仍低于 NLI Baseline。目标是超过 NLI Baseline（0.8392）。

#### 改进路径

**路径 1：修复数据质量（先决条件）**

质量差的 hard_neg（语法错误 / 语义伪正样本）会引入错误梯度，导致模型退化。在解决问题 1a/1b 之前，数据量再大也很难超越 NLI Baseline。

**路径 2：使用所有 10 种方法的 Hard Neg（多样性）**

当前只用了 `direct_negation_attack` 一种方法。不同方法覆盖不同语言现象：
- 数字/实体替换：适合事实型 STS 对
- 否定攻击：适合推断型 STS 对
- 逻辑关系改写：适合复杂从句 STS 对

多方法混合数据能让模型在更广的语言边界上学到有意义的距离函数，有助于 MTEB 泛化。

**路径 3：Curriculum Learning（效果最稳定）**

```
Phase 1: NLI 全量数据 + 原始 neg → MNRL（通用语义对齐）
Phase 2: 所有方法 Hard Neg → TripletLoss（边界精调）
```

NLI 数据（数十万到百万级）保证了通用表示质量，Hard Neg 数据（质量更高后可达数万）负责精调困难边界。

**路径 4：增加训练轮次**

当前 `num_train_epochs=1`。对于较小规模的 hard neg 数据，适当增加到 2–3 epochs（注意 warmup 比例相应调整）可以提升收敛效果。

**路径 5：更大的有效 batch size**

MNRL 的对比效果依赖 batch 多样性，更大的 batch size 效果更好。可使用 `CachedMultipleNegativesRankingLoss` 以 gradient cache 技术支持等效更大 batch，而不增加显存：

```python
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)
```

---

## 行动优先级建议

| 优先级 | 问题 | 改进动作 | 预期收益 |
|--------|------|---------|---------|
| **P0** | 1a 语法错误 | 添加缩写查找表 + 后处理过滤器 | 消除脏数据污染 |
| **P0** | 1b 语义伪正样本 | 加入嵌入相似度过滤（阈值 0.90）；从 LLM 路径移除 `premise_disruption` | 提升 hard_neg 有效率 |
| **P1** | 2 损失函数 | 加入 Curriculum：Phase 1 MNRL + Phase 2 TripletLoss（margin=0.15） | 提高困难边界学习效果 |
| **P1** | 3b MTEB 提升 | 引入全部 10 种方法的 hard_neg 数据 | 多样化覆盖提升泛化 |
| **P2** | 3a MTEB 接口 | 补全 7 个 STS 任务的完整评估 | 确认与榜单对齐 |
| **P2** | 2 损失函数 | 尝试 `CachedMNRL` 提升有效 batch size | 增量提升 |
