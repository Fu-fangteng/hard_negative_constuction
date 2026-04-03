
---

# 二阶段：规范化模块化的困难负样本训练集构造 Pipeline 设计说明

## 1. 项目目标

以 nli_for_simcse 的三元组数据（{anchor, positive, negative}）为基础，自动化构建高质量、可控的困难负样本（hard negative），扩展为四元组 {anchor, positive, negative, hard_negative}，为对比学习等任务提供更具挑战性的训练集。

## 2. 总体流程

1. **数据预处理**  
   - 输入：原始 nli 三元组数据（jsonl）
   - 输出：标准化三元组（id, anchor, pos, neg）

2. **样本采样（可选）**  
   - 支持对原始数据进行采样或筛选，便于实验和调试。

3. **实体识别与特征抽取**  
   - 对 positive 句子进行实体/特征识别，支持两种方式：
     - 正则表达式（规则法）
     - 开源 LLM（如 Qwen）模型识别
   - 仅允许每个句子应用一种构造方法。

4. **困难负样本构造方法实现**  
   - 参考 hard_neg_construction_method.md 中的方法，逐一实现。
   - 每种方法均需支持两种实体识别方式（Regular/LLM）。
   - 典型方法如：直接否定、程度副词、数值变换、实体代词替换等。

5. **困难负样本生成**  
   - 对 positive 应用选定方法，生成 hard_negative。
   - 记录方法、替换日志、统计信息。

6. **数据结构与输出组织**  
   - 目录结构规范，便于后续扩展和复现。
   - 每种方法、每种识别方式分别存储结果、统计和日志。

7. **测试与训练脚本（预留）**  
   - 当前阶段仅实现训练集构建，inference 和 train 相关内容单独归档。

## 3. 目录与数据结构规范

```
Data/
├── original_data/
│   └── nli_train.jsonl
├── preprocessed_data/
│   └── preprocessed_data.json  # {id, anchor, pos, neg}
├── processed_data/
│   └── <method_name>/
│       └── difference.md       # 两种方法构造差异说明
│       └── LLM/
│       │   ├── method_stat.json
│       │   ├── constructed_data.json  # {id, pos, neg, hard_neg}
│       │   └── construction_log
│       └── Regular/
│           ├── method_stat.json
│           ├── constructed_data.json
│           └── construction_log
│   └── ...（每种方法同上结构）
```

## 4. 关键实现要点

- **实体识别与替换**  
  - 每种方法均需实现正则与 LLM 两种 pipeline，便于对比。
  - construction_log 需详细记录识别、替换、失败等统计。

- **方法选择与应用**  
  - 每个 positive 句子仅应用一种方法，方法选择可配置或自动分配。

- **可扩展性**  
  - 新增方法时，按上述目录规范扩展即可。
  - 支持后续加入更多识别模型或构造策略。

- **日志与统计**  
  - 每次构造需输出 method_stat.json（方法统计）、construction_log（详细日志）、difference.md（两种方法差异分析）。

## 5. 后续扩展建议

- **inference/ 目录**：将评估、推理相关脚本和结果单独归档，便于管理。
- **train/ 目录**：后续训练脚本、配置等单独管理。
- **方法自动选择**：可根据特征自动分配最优构造方法。

## 6. 参考与约定

- 具体构造方法详见 hard_neg_construction_method.md。
- 代码实现风格、接口、数据格式等，优先参考现有 pipeline 设计。

