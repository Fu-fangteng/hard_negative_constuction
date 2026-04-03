# Stage 1 vs Stage 2 识别与构造对比报告

> 基于已有运行数据统计，不含新代码。  
> 生成时间：2026-04-04

---

## 一、数据规模对比

| 维度 | Stage 1 | Stage 2 |
|---|---|---|
| 数据来源 | STS 数据集（top-100 高相似正样本对） | NLI nli_for_simcse（1000 条采样） |
| 输入格式 | `(text1, text2, score)` | `(anchor, pos, neg)` |
| 输出格式 | `(text1, text2, text3)` | `(anchor, pos, neg, hard_neg)` |
| 总样本数 | **100** | **1000** |
| 成功生成硬负样本 | **99** (99%) | **1000** (100%) |
| 识别路径 | regex + spaCy（可选）+ LLM（可选） | regex + 正则实体识别 + spaCy（可选）+ LLM（可选） |

---

## 二、特征识别统计

> Stage 1 与 Stage 2 均未加载 spaCy (`en_core_web_sm`)，以下数据为**纯规则识别**结果。

### 2.1 Stage 1 — 特征覆盖率（100 samples）

| 特征字段 | 有该特征的样本数 | 识别到的 token 总数 |
|---|---|---|
| numbers | 21 / 100 (21%) | 37 |
| logic_words | 4 / 100 (4%) | 5 |
| sequence_words | 3 / 100 (3%) | 3 |
| degree_words | 8 / 100 (8%) | 10 |
| negations | 9 / 100 (9%) | 9 |
| pronouns | 14 / 100 (14%) | 15 |
| **entities** | **0 / 100 (0%)** | **0** |
| subject_candidates | 0 / 100 (0%) | 0 |
| object_candidates | 0 / 100 (0%) | 0 |

> **注**：Stage 1 entities 全为 0，因为 spaCy 未安装，而 `_regex_extract()` 本身不提取实体。`entity_pronoun_substitution` 能生效完全依赖代词替换（14 个有代词的样本）。

### 2.2 Stage 2 — 方法特征覆盖率（1000 samples，Regular 路径）

| 方法 | 有特征的样本数 | 特征总数 | 覆盖率 |
|---|---|---|---|
| numeric_metric_transform | 23 | 24 | 2.3% |
| **entity_pronoun_substitution** | **193** | **215** | **19.3%** |
| scope_degree_scaling | 74 | 80 | 7.4% |
| direct_negation_attack | 939 | 939 | 93.9% |
| double_negation_attack | 61 | 61 | 6.1% |
| logical_operator_rewrite | 51 | 52 | 5.1% |
| role_swap | 0 | 0 | 0% |
| temporal_causal_inversion | 18 | 19 | 1.8% |
| concept_hierarchy_shift | 0 | 0 | 0% |
| premise_disruption | 1000 | 1000 | 100% |

> **注**：`entity_pronoun_substitution` Stage 2 的特征计数来自 pronouns + entities（正则识别），比 Stage 1 提升显著。  
> `concept_hierarchy_shift` 特征计数为 0（依赖 entities），但实际构造成功 39 例，因为该方法内置词表不依赖特征字段（见第三节）。

---

## 三、构造方法成功率对比

### 3.1 各方法成功率

| 方法 | Stage 1（100 samples） | Stage 2（1000 samples，Regular） |
|---|---|---|
| numeric_metric_transform | 20/100 = **20.0%** | 16/1000 = **1.6%** |
| entity_pronoun_substitution | 13/100 = **13.0%** | 193/1000 = **19.3%** |
| scope_degree_scaling | 5/100 = **5.0%** | 75/1000 = **7.5%** |
| direct_negation_attack | 59/100 = **59.0%**¹ | 919/1000 = **91.9%** |
| double_negation_attack | 2/100 = **2.0%** | 81/1000 = **8.1%** |
| logical_operator_rewrite | 0/100 = 0% | 52/1000 = **5.2%** |
| role_swap | 0/100 = 0% | 0/1000 = **0%** |
| temporal_causal_inversion | 0/100 = 0%² | 19/1000 = **1.9%** |
| concept_hierarchy_shift | 0/100 = 0%² | 39/1000 = **3.9%** |
| premise_disruption | 0/100 = 0%² | 1000/1000 = **100%** |

> ¹ Stage 1 成功率指方法**实际被调度并成功**的比例（在 auto 模式下按优先级分配，不是所有方法都会尝试每条样本）  
> ² Stage 1 的 temporal / concept / premise 实际成功率可能 > 0，但因为 direct_negation_attack 优先级更高，在 auto 模式下抢先匹配，这些方法几乎没有机会被调用到

### 3.2 最终数据集方法分布

**Stage 1 final_dataset（99 samples）：**

| 方法 | 使用次数 | 占比 |
|---|---|---|
| direct_negation_attack | 59 | 59.6% |
| numeric_metric_transform | 20 | 20.2% |
| entity_pronoun_substitution | 13 | 13.1% |
| scope_degree_scaling | 5 | 5.1% |
| double_negation_attack | 2 | 2.0% |
| 其余 5 种 | 0 | 0% |

**Stage 2 final_dataset（1000 samples，去重后各取第一个成功方法）：**

| 方法 | 使用次数 | 占比 |
|---|---|---|
| direct_negation_attack | 692 | 69.2% |
| entity_pronoun_substitution | 191 | 19.1% |
| scope_degree_scaling | 57 | 5.7% |
| double_negation_attack | 44 | 4.4% |
| numeric_metric_transform | 16 | 1.6% |
| 其余 5 种 | 0 | 0%³ |

> ³ logical_operator_rewrite/role_swap/temporal/concept/premise 在 final_dataset 中贡献为 0，因为前 5 种方法已覆盖全部 1000 条样本（direct_negation_attack 91.9% + 前几种方法叠加 = 100%）。

---

## 四、实体识别专项对比

实体识别是本次 Stage 2 重构的核心修复点。

| 对比维度 | Stage 1 | Stage 2（修复后） |
|---|---|---|
| 实体识别依赖 | 仅 spaCy（未安装则为 0） | **正则启发式** + spaCy（可选） |
| entities 覆盖样本数 | **0 / 100 (0%)** | **193 / 1000 (19.3%)** |
| entity_pronoun_substitution 成功率 | 13% | **19.3%** |
| 实体来源 | — | 称谓+姓名 / 大写词序列 / 全大写缩写 |

**Stage 2 entity_pronoun_substitution 成功案例（前 5 例）：**

| 输入 pos | 输出 hard_neg | 变化 |
|---|---|---|
| ...knife against the man's bare throat as **he** was pushing... | ...as **she** was pushing... | he → she |
| A few found purchase, but most of **them** were swept away... | ...most of **it** were swept away... | them → it |
| The Kal swung **his** club high. | The Kal swung **her** club high. | his → her |
| ...you have to mail **it** back to the store... | ...mail **they** back to the store... | it → they |
| The car has held up well but when I gifted **it** to my son... | ...gifted **they** to my son... | it → they |

---

## 五、关键问题与观察

### 5.1 两阶段共同弱点

| 问题 | 原因 | 影响方法 |
|---|---|---|
| `role_swap` 永远 0% | 需要 spaCy 的 nsubj/dobj 依存关系 | role_swap |
| `concept_hierarchy_shift` 覆盖率极低 | 内置词表只有 12 个词（apple/dog/car 等） | concept_hierarchy_shift |
| `numeric_metric_transform` Stage 2 仅 1.6% | NLI 数据本身含数字的句子少 | numeric_metric_transform |

### 5.2 两阶段差异解释

**entity_pronoun_substitution Stage 2 (19.3%) > Stage 1 (13%)** 的主要原因：
- Stage 1 仅靠代词替换，14/100 = 14% 基线
- Stage 2 新增正则实体识别，在代词基础上叠加了实体替换

**direct_negation_attack Stage 2 (91.9%) > Stage 1 (59%)** 的主要原因：
- Stage 1 是 auto 模式下实际调用并成功的次数（其他方法先被调用）
- Stage 2 每条样本独立运行该方法，NLI 数据本身否定句比例低，约 6.1% 已含否定（不可再施加直接否定），故约 93.9% 的样本可以加否定

### 5.3 Stage 2 final_dataset 分布问题

当前的 `aggregate_final_dataset` 按 ALL_METHODS 顺序取第一个成功方法，导致：
- `premise_disruption`（100% 成功率）在队列末尾，实际贡献 0 条
- `direct_negation_attack` 吃掉 69.2% 的样本
- 方法多样性不足，最终数据集中 5 种方法贡献 0

**建议**：可考虑按样本特征分配最优方法，而不是"先到先得"策略。

---

## 六、总结

| 指标 | Stage 1 | Stage 2 |
|---|---|---|
| 样本量 | 100 | 1000 |
| 最终生成成功率 | 99% | **100%** |
| 实体识别覆盖 | 0%（spaCy 缺失） | **19.3%**（正则兜底） |
| 有效方法数量 | 5 种 | 5 种 |
| 方法多样性（final） | 有限，direct_neg 主导 | 有限，direct_neg 主导 |
| 评估 Gap（S1-S2）均值 | **0.71**（angle_emb） | — |
| 评估有效率（S2<S1） | **97.98%** | — |

Stage 2 在规模（10×）和实体识别覆盖率上有显著提升；两阶段共同的瓶颈在于 `role_swap` 依赖 spaCy、`concept_hierarchy_shift` 词表过小，以及 final_dataset 的方法分布高度集中于 `direct_negation_attack`。
