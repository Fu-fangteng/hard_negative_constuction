# Stage 1 vs Stage 2 识别与构造对比报告

> 基于已有运行数据统计。Stage 2 数据含 spaCy (`en_core_web_sm`) 加持。  
> 最后更新：2026-04-04

---

## 一、数据规模对比

| 维度 | Stage 1 | Stage 2 |
|---|---|---|
| 数据来源 | STS 数据集（top-100 高相似正样本对） | NLI nli_for_simcse（1000 条采样） |
| 输入格式 | `(text1, text2, score)` | `(anchor, pos, neg)` |
| 输出格式 | `(text1, text2, text3)` | `(anchor, pos, neg, hard_neg)` |
| 总样本数 | **100** | **1000** |
| 成功生成硬负样本 | **99** (99%) | **1000** (100%) |
| spaCy 可用 | ❌ 未安装 | ✅ en_core_web_sm |
| 评估 backend | angle_emb | — |

---

## 二、特征识别对比

### 2.1 Stage 1 — 特征覆盖率（100 samples，无 spaCy）

| 特征字段 | 有该特征的样本数 | token 总数 |
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

> entities / subject / object 全为 0：spaCy 未安装，`_regex_extract()` 本身不提取实体。

### 2.2 Stage 2 — 方法特征覆盖率（1000 samples，Regular 路径，含 spaCy）

| 方法 | 有特征的样本数 | 覆盖率 | token 总数 |
|---|---|---|---|
| numeric_metric_transform | 23 | 2.3% | 24 |
| **entity_pronoun_substitution** | **529** | **52.9%** | **796** |
| scope_degree_scaling | 74 | 7.4% | 80 |
| direct_negation_attack | 939 | 93.9% | 939 |
| double_negation_attack | 61 | 6.1% | 61 |
| logical_operator_rewrite | 51 | 5.1% | 52 |
| **role_swap** | **670** | **67.0%** | **736** |
| temporal_causal_inversion | 18 | 1.8% | 19 |
| concept_hierarchy_shift | 410 | 41.0% | 581 |
| premise_disruption | 1000 | 100% | 1000 |

> spaCy 带来的核心提升：`entity_pronoun_substitution` 特征覆盖 19.3% → **52.9%**；`role_swap` 特征覆盖 0% → **67.0%**；`concept_hierarchy_shift` 0% → **41.0%**。

---

## 三、构造方法成功率对比

| 方法 | Stage 1（100 samples，无 spaCy） | Stage 2（1000 samples，含 spaCy） | 变化 |
|---|---|---|---|
| numeric_metric_transform | 20 / 100 = 20.0%¹ | 16 / 1000 = **1.6%** | ↓ NLI 数据含数字少 |
| entity_pronoun_substitution | 13 / 100 = 13.0% | 203 / 1000 = **20.3%** | ↑ spaCy 识别实体 |
| scope_degree_scaling | 5 / 100 = 5.0% | 75 / 1000 = **7.5%** | ↑ 略 |
| direct_negation_attack | 59 / 100 = 59.0%¹ | 919 / 1000 = **91.9%** | ↑ 独立运行非 auto |
| double_negation_attack | 2 / 100 = 2.0% | 81 / 1000 = **8.1%** | ↑ |
| logical_operator_rewrite | 0 / 100 = 0%¹ | 52 / 1000 = **5.2%** | ↑ 独立运行 |
| **role_swap** | **0 / 100 = 0%** | **664 / 1000 = 66.4%** | ↑↑ spaCy 依存句法 |
| temporal_causal_inversion | 0 / 100 = 0%¹ | 19 / 1000 = **1.9%** | ↑ 独立运行 |
| concept_hierarchy_shift | 0 / 100 = 0%¹ | 39 / 1000 = **3.9%** | ↑ 独立运行 |
| premise_disruption | 0 / 100 = 0%¹ | 1000 / 1000 = **100%** | ↑ 独立运行 |

> ¹ Stage 1 为 auto 模式，按优先级调度：`direct_negation_attack` 提前抢占大多数样本，后续方法几乎没有机会被调用。Stage 2 每条样本对所有方法**独立运行**，不互相抢占。

**最大收益：`role_swap` 0% → 66.4%**，完全依赖 spaCy 的 nsubj/dobj 依存分析。

---

## 四、最终数据集方法分布

### Stage 1 final_dataset（99 samples，auto 模式）

| 方法 | 使用次数 | 占比 |
|---|---|---|
| direct_negation_attack | 59 | 59.6% |
| numeric_metric_transform | 20 | 20.2% |
| entity_pronoun_substitution | 13 | 13.1% |
| scope_degree_scaling | 5 | 5.1% |
| double_negation_attack | 2 | 2.0% |
| 其余 5 种 | 0 | 0% |

### Stage 2 final_dataset（1000 samples，first-method-wins 去重）

| 方法 | 使用次数 | 占比 |
|---|---|---|
| direct_negation_attack | 685 | 68.5% |
| entity_pronoun_substitution | 200 | 20.0% |
| scope_degree_scaling | 57 | 5.7% |
| double_negation_attack | 42 | 4.2% |
| numeric_metric_transform | 16 | 1.6% |
| role_swap / logical / temporal / concept / premise | 0 | 0% |

> **注**：`role_swap` 单方法成功 664/1000，但在 final_dataset 中贡献为 0，原因是 `aggregate_final_dataset` 按 ALL_METHODS 顺序取第一成功方法。`direct_negation_attack`（第 4 位）已将 685 条抢占，前三种方法合计覆盖剩余 315 条，`role_swap`（第 7 位）无样本可用。

---

## 五、spaCy 带来的提升汇总

| 指标 | 无 spaCy | 有 spaCy | 提升 |
|---|---|---|---|
| entity_pronoun_sub 特征覆盖 | 19.3% | **52.9%** | +33.6pp |
| entity_pronoun_sub 成功率 | 19.3% | **20.3%** | +1pp |
| role_swap 特征覆盖 | 0% | **67.0%** | +67pp |
| role_swap 成功率 | 0% | **66.4%** | +66.4pp |
| concept_hierarchy_shift 特征覆盖 | 0% | **41.0%** | +41pp |
| concept_hierarchy_shift 成功率 | 3.9% | **3.9%** | — (内置词表决定上限) |

> `concept_hierarchy_shift` 特征覆盖大幅提升（实体字段从 0 → 41%），但构造成功率不变，因为该方法实际上使用内置的 12 词词表而非 entities 字段，词表是真正瓶颈。

---

## 六、评估指标（Stage 1，angle_emb）

| 指标 | 值 |
|---|---|
| 样本数 | 99 |
| S1 均值（text1-text2） | 4.831 |
| S2 均值（text1-text3） | 4.119 |
| Gap 均值（S1-S2） | **0.712** |
| Gap 中位数 | 0.604 |
| 有效率（S2 < S1） | **97.98%** |

| 方法 | 样本数 | Gap 均值 |
|---|---|---|
| direct_negation_attack | 59 | **1.045** |
| double_negation_attack | 2 | 0.381 |
| numeric_metric_transform | 20 | 0.321 |
| entity_pronoun_substitution | 13 | 0.101 |
| scope_degree_scaling | 5 | 0.065 |

> Stage 2 尚未运行 angle_emb 评估。

---

## 七、问题与建议

| 问题 | 影响 | 建议 |
|---|---|---|
| `concept_hierarchy_shift` 词表只有 12 词 | 构造成功率 3.9% 与词表挂钩，与 spaCy 无关 | 扩充词表或接入 WordNet |
| `numeric_metric_transform` NLI 数据含数字率仅 2.3% | 成功率 1.6%，数据集特性决定 | 对含数字的样本加权采样 |
| final_dataset 方法分布高度集中 | `direct_negation_attack` 占 68.5%，多样性不足 | 改为按特征分配最优方法，而非 first-method-wins |
| `role_swap` 语法质量 | 64% 成功，但主宾互换可能产生语法错误 | 可加后处理过滤明显语法错误的输出 |
