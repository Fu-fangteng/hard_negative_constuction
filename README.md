# STS 困难负样本构造（Hard Negatives）

这个项目用于在语义文本相似度（STS, Semantic Textual Similarity）数据集上构造“困难负样本”（Hard Negatives）。
给定原始正样本对 `(text1, text2, score)`，通过对 `text2` 进行可控扰动生成 `text3`，使得相似度从 `S1=sim(text1,text2)` 明显下降到 `S2=sim(text1,text3)`。

## 1. 数据格式约定

项目全链路统一使用如下字段（类型：`id/text1/text2` 为字符串，`score` 为浮点数）：

- `id`: 样本唯一标识（如数据里没有则自动生成）
- `text1`: 句子 1
- `text2`: 句子 2（后续要被扰动的对象）
- `score`: STS 参考/标注分数（Ground Truth，用于后续偏移分析）

输入文件支持：

- `.jsonl`: 每行一个 JSON 对象
- `.json`: list[object]
- `.csv`: 列包含 `id,text1,text2,score`（或至少包含 text1/text2/score 的等价字段）

## 2. 七个模块（体系化实现）

### 模块 1：数据预处理 `src/data_utils.py`

- 读取原始数据并清洗空白（避免无意义空格差异）
- 统一映射到规范结构：`id/text1/text2/score`
- 导出为 `.json` / `.csv`

### 模块 2：数据筛选 `src/sampler.py`

- 根据 `score` 选择前 `top_k` 条高相似正样本对
- 主要用于构造实验子集（默认 `k=100`，可改）

### 模块 3：LLM 驱动引擎 `src/llm_engine.py`（可选）

- 预留本地加载开源模型的能力（如 Qwen 系列）
- 本轮流水线默认未启用 LLM 抽取（`llm_engine=None` 可直接跑通规则版）

### 模块 4：特征识别与自动格式化 `src/formatter.py`

核心目标：对每条 `text2` 提取可用于“构造方法选择/执行”的特征，并输出：

- `formatted_data.json`: `id/text1/text2/score + features + methods_available`
- `methods_stat.json`: 每个方法的可用性标记（0/1）+ `feature_count`

特征来源（由强到弱）：

- 规则抽取（正则）：数值、逻辑词、顺序词、程度副词、否定词、代词等
- 可选 spaCy：实体/主语候选/宾语候选（若模型没装则自动降级）
- 可选 LLM：通过 `src/prompts.py` 提供的严格 JSON Prompt 抽取结构化特征

### 模块 5：构造方法库 `src/constructors.py`

实现 10 种困难负样本生成函数（失败返回 `None`，成功返回 `text3`）：

1. `numeric_metric_transform`
2. `entity_pronoun_substitution`
3. `scope_degree_scaling`
4. `direct_negation_attack`
5. `double_negation_attack`
6. `logical_operator_rewrite`
7. `role_swap`
8. `temporal_causal_inversion`
9. `concept_hierarchy_shift`
10. `premise_disruption`

注：当前版本以规则为主（对“可用性”进行兜底），LLM 改写能力保留在后续扩展中。

### 模块 6：核心调度 `src/main_generator.py`

- 从 `formatted_data` 中读取特征和方法可用性
- 支持单方法或组合（最多 1~3 个方法，按固定优先级依次尝试）
- 校验生成结果：`text3` 必须非空且与 `text2` 不完全相同
- 输出字段：`id/text1/text2/text3/score + methods_used`

### 模块 7：评估与指标分析 `src/evaluator.py`

基于 `angle_emb` 计算句向量余弦相似度：

- `S1 = sim(text1, text2)`
- `S2 = sim(text1, text3)`
- `Gap = S1 - S2`（Gap 越大代表 `text3` 越“难”）

并计算：

- `validity_ratio_S2_lt_S1`: `S2 < S1` 的比例
- `method_contribution`: 以 `methods_used` 为集合归因（方法出现则纳入该方法的 Gap 统计）
- `GT_offset_stats`: `S2 - score(ground truth)` 的 mean/var/median
- Gap/S1/S2 的 `mean/var/median`

可视化（matplotlib，可缺省）：

- Gap 直方图
- S1 vs S2 散点图
- S1/S2 箱线图

## 3. 如何运行（端到端）

运行脚本（不做评估，仅生成数据文件）：

```bash
python3 scripts/run_pipeline.py --input data/test.jsonl --out_dir outputs/run_1 --k 100
```

如果启用评估（需要 `angle-emb` 依赖正确安装）：

```bash
python3 scripts/run_pipeline.py --input data/test.jsonl --out_dir outputs/run_1 --k 100 --evaluate
```

指定方法组合（可选）：

```bash
python3 scripts/run_pipeline.py --input data/test.jsonl --out_dir outputs/run_1 --k 100 --methods direct_negation_attack,entity_pronoun_substitution
```

`--methods auto` 表示根据 `methods_available` 自动选择（默认行为）。

## 4. 关键输出文件

在 `--out_dir` 下通常包括：

- `topk_positives.json/.csv`：前 `k` 条正样本对
- `formatted_data.json`：特征 + 方法可用性
- `methods_stat.json`：方法可用性统计
- `final_dataset.jsonl/.csv`：最终困难负样本数据集
- （可选）`evaluation_report.json`：评估指标汇总 + 可视化路径

