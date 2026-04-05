# 困难负样本构造方法详解

本文档描述项目实现的全部 10 种困难负样本构造方法，分为四大模块。

---

## 模块一：局部事实置换

### 1. `numeric_metric_transform` — 数值与度量变换

**原理**：改变句子中的数字、百分比、货币、度量单位，使事实陈述发生量化偏移。

**特征依赖**：`numbers`（正则提取）

**实现**：
- 识别所有数字 token（整数、浮点数、百分比）
- 随机乘以系数（×0.5 ~ ×2.0）并取整
- 单数/复数同步调整（如 "1 student" → "3 students"）

**示例**：
```
原文：The temperature rose from 20°C to 30°C in 5 minutes.
构造：The temperature rose from 14°C to 30°C in 5 minutes.
```

**Stage 2 成功率**：1.6%（NLI 数据含数字率仅 2.3%，数据特性决定上限）

---

### 2. `entity_pronoun_substitution` — 实体/指代置换

**原理**：替换人名、地名、组织名，或修改代词指向，破坏实体级事实。

**特征依赖**：`entities`（spaCy NER 或 regex 启发式）、`pronouns`（正则）

**实现**：
- Step 1（代词）：he↔she、his↔her、him↔her 等双向互换
- Step 2（实体）：在预定义替换表（~80 条目）中查找并替换
  - 人名：John→Michael, Mary→Sarah, David→James, ...
  - 地名：Paris→London, New York→Los Angeles, Tokyo→Berlin, ...
  - 机构：Google→Apple, Harvard→Yale, NASA→ESA, ...
- 保留原词的大小写风格（全大写/首字母大写/小写）

**示例**：
```
原文：Michael went to Paris and met with the mayor.
构造：John went to London and met with the mayor.
```

**Stage 2 成功率**：20.3%（spaCy 实体覆盖率 52.9%，无 spaCy 时 regex 补充）

---

### 3. `scope_degree_scaling` — 范围与程度缩放

**原理**：修改量词（all/some/most）和程度副词（very/extremely/slightly），改变陈述的范围或强度。

**特征依赖**：`degree_words`、`logic_words`（正则提取）

**实现**：
- 量词替换：all↔some、most↔few、every↔some、many↔few
- 模态词替换：must↔might、always↔sometimes、never↔rarely
- 程度副词替换：very/extremely → slightly、completely → partially

**示例**：
```
原文：All students must complete the assignment.
构造：Some students might complete the assignment.
```

**Stage 2 成功率**：7.5%

---

## 模块二：极性与逻辑反转

### 4. `direct_negation_attack` — 直接否定攻击

**原理**：在谓语动词处添加否定词（not/never），直接翻转命题真值。

**特征依赖**：无特殊特征（通用方法，对所有句子可用）

**实现**：
- 识别助动词（is/are/was/were/can/will/do/does/did/has/have）插入 not
- 对无助动词的简单句加 "do/does/did not"
- 避免双重否定（已含 not 的句子跳过）

**示例**：
```
原文：The cat is sleeping on the couch.
构造：The cat is not sleeping on the couch.
```

**Stage 2 成功率**：91.9%（覆盖率最高，是 final_dataset 的主力方法）

---

### 5. `double_negation_attack` — 多重否定攻击

**原理**：在已有否定词处加入双重否定，使语义逻辑变复杂而偏离原意。

**特征依赖**：`negations`（正则提取 not/never/no/neither/nor 等）

**实现**：
- 识别现有否定词，在其前加 "not" 形成双重否定
- 或替换 "never" → "not always"、"no" → "not every"

**示例**：
```
原文：She never goes to the gym.
构造：She does not never go to the gym.
```

**Stage 2 成功率**：8.1%

---

### 6. `logical_operator_rewrite` — 逻辑算子改写

**原理**：互换因果、转折、并列、条件关系的连接词，破坏逻辑结构。

**特征依赖**：`logic_words`（正则提取 because/but/and/if/although 等）

**实现**：
- because → although、since → but
- and → but、but → and
- if → even if、unless → if

**示例**：
```
原文：She passed the exam because she studied hard.
构造：She passed the exam although she studied hard.
```

**Stage 2 成功率**：5.2%

---

## 模块三：结构与时序重组

### 7. `role_swap` — 角色互换（主宾对调）

**原理**：交换句子的主语和宾语，使施动者/受动者关系完全颠倒。

**特征依赖**：`subject_candidates` + `object_candidates`（**依赖 spaCy 依存分析** nsubj/dobj）

**实现**：
- 使用 spaCy 解析句子，找到 nsubj（主语）和 dobj（宾语）
- 将主语和宾语位置互换，保持其他词不变
- 若句子含被动结构跳过

**示例**：
```
原文：The dog chased the cat.
构造：The cat chased the dog.
```

**注意**：完全依赖 spaCy，未安装时成功率为 0%。安装后 Stage 2 成功率 **66.4%**。

---

### 8. `temporal_causal_inversion` — 时序与因果倒置

**原理**：调换事件的先后顺序或因果关系，使时间线或逻辑链错误。

**特征依赖**：`sequence_words`（正则提取 before/after/then/first/finally 等）

**实现**：
- before ↔ after、first ↔ last、then → before
- 识别逗号分隔的两个子句，交换顺序

**示例**：
```
原文：She ate dinner before watching TV.
构造：She ate dinner after watching TV.
```

**Stage 2 成功率**：1.9%（NLI 数据时序词较少）

---

## 模块四：知识与常识偏置

### 9. `concept_hierarchy_shift` — 概念层级偏移

**原理**：用上位词、下位词或同级词替换关键概念，使语义层级发生偏移。

**特征依赖**：`entities` 或内置 12 词词表（animal/vehicle/food/sport/music/art/science/technology/nature/building/clothing/tool）

**实现**：
- 检查句子是否包含内置词表中的词
- 替换为同级概念（如 animal→vehicle、food→sport）
- 若有实体，尝试上位词替换（如 dog→animal）

**示例**：
```
原文：The athlete trained for the sport competition.
构造：The athlete trained for the music competition.
```

**限制**：词表仅 12 词，成功率 3.9%，词表是真正瓶颈而非 spaCy。

---

### 10. `premise_disruption` — 前提破坏

**原理**：插入与句子前提矛盾的短语，使整体语义自相矛盾。

**特征依赖**：无（通用兜底方法，100% 覆盖率）

**实现**：
- 在句首插入对立前提短语，如：
  - "Contrary to what was stated, ..."
  - "Despite the opposite being true, ..."
  - "Although the evidence suggests otherwise, ..."
- 固定模板，不依赖任何特征识别

**示例**：
```
原文：The team won the championship.
构造：Contrary to what was stated, the team won the championship.
```

**Stage 2 成功率**：100%（所有样本都能生成，但语义扰动强度相对较弱）

---

## 方法对比总览

| # | 方法 | spaCy 依赖 | Stage 2 成功率 | 主要限制 |
|---|---|---|---|---|
| 1 | numeric_metric_transform | ❌ | 1.6% | NLI 数据含数字少 |
| 2 | entity_pronoun_substitution | 可选 | 20.3% | 替换表覆盖范围 |
| 3 | scope_degree_scaling | ❌ | 7.5% | 程度词密度 |
| 4 | direct_negation_attack | ❌ | 91.9% | — |
| 5 | double_negation_attack | ❌ | 8.1% | 需原文含否定词 |
| 6 | logical_operator_rewrite | ❌ | 5.2% | 需含逻辑连接词 |
| 7 | role_swap | **✅ 必须** | 66.4% | spaCy 未安装则 0% |
| 8 | temporal_causal_inversion | ❌ | 1.9% | NLI 数据时序词少 |
| 9 | concept_hierarchy_shift | 可选 | 3.9% | 词表仅 12 词 |
| 10 | premise_disruption | ❌ | 100% | 扰动强度弱 |

> 按 Gap 强度排序（Stage 1 angle_emb 评估）：`direct_negation_attack` (1.045) > `double_negation_attack` (0.381) > `numeric_metric_transform` (0.321) > `entity_pronoun_substitution` (0.101) > `scope_degree_scaling` (0.065)

---

## 后续改进建议

| 问题 | 建议 |
|---|---|
| `concept_hierarchy_shift` 词表仅 12 词 | 接入 WordNet hypernym/hyponym |
| `role_swap` 主宾互换可能产生语法错误 | 加后处理过滤明显语法错误的输出 |
| final_dataset 中 `direct_negation_attack` 占比 68.5%，多样性不足 | 改为按特征分配最优方法，而非 first-method-wins |
| `numeric_metric_transform` 在 NLI 数据上成功率 1.6% | 对含数字的样本加权采样 |
