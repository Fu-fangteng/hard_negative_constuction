# Stage 1 快速上手指南

本文档说明如何运行 Stage 1 流水线（STS 数据集困难负样本构造）。

---

## 1. 环境准备

```bash
conda activate rl
pip install pandas angle-emb spacy
python -m spacy download en_core_web_sm   # 可选，缺失时 role_swap 等方法跳过
```

---

## 2. 数据格式

输入文件支持 `.jsonl`、`.json`、`.csv`，至少包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 样本唯一标识（可自动生成） |
| `text1` | str | 句子 1 |
| `text2` | str | 句子 2（将被扰动生成 text3） |
| `score` | float | STS 标注相似度分数 |

---

## 3. 端到端运行

```bash
# 基本（100 条，auto 方法选择）
python stage1/run_pipeline.py \
    --input data/raw/test.jsonl \
    --out_dir data/stage1/run_1 \
    --k 100

# 启用评估（需要 angle-emb）
python stage1/run_pipeline.py \
    --input data/raw/test.jsonl \
    --out_dir data/stage1/run_1 \
    --k 100 --evaluate

# 指定构造方法
python stage1/run_pipeline.py \
    --input data/raw/test.jsonl \
    --out_dir data/stage1/run_1 \
    --k 100 \
    --methods direct_negation_attack,entity_pronoun_substitution
```

参数 `--methods auto` 表示根据 `methods_available` 自动选择（默认行为）。

---

## 4. 输出文件

运行后 `--out_dir` 下包含：

| 文件 | 说明 |
|---|---|
| `topk_positives.json/.csv` | 前 k 条正样本对 |
| `formatted_data.json` | 特征 + 方法可用性 |
| `methods_stat.json` | 方法可用性统计 |
| `final_dataset.jsonl/.csv` | 最终困难负样本数据集 |
| `evaluation_report.json` | 评估指标（`--evaluate` 时生成） |

---

## 5. 核心模块说明

| 模块 | 文件 | 职责 |
|---|---|---|
| 数据预处理 | `stage1/data_utils.py` | 读取清洗，统一映射到 id/text1/text2/score |
| 数据筛选 | `stage1/sampler.py` | top-k 高相似正样本对 |
| LLM 引擎 | `stage1/llm_engine.py` | 本地 LLM 封装（可选，默认 None） |
| 特征识别 | `stage1/formatter.py` | regex + spaCy + LLM，输出 features + methods_available |
| 构造方法库 | `stage1/constructors.py` | 10 种方法，失败返回 None |
| 调度器 | `stage1/main_generator.py` | 按方法优先级生成 text3 |
| 评估 | `stage1/evaluator.py` | S1/S2/Gap 计算，可视化 |

---

## 6. 特征识别层级

```
text2
  ├─ 规则（正则）：numbers / logic_words / sequence_words / degree_words / negations / pronouns
  ├─ spaCy（可选）：entities / subject_candidates / object_candidates（nsubj/dobj 依存分析）
  └─ LLM（可选）：通过 prompts.py 的结构化 JSON prompt 抽取
```

spaCy 未安装时自动降级，仅规则特征可用；`role_swap` 等依赖 nsubj/dobj 的方法将跳过。

---

## 7. 评估指标

| 指标 | 说明 |
|---|---|
| S1 = sim(text1, text2) | 正样本相似度（angle_emb 余弦） |
| S2 = sim(text1, text3) | 困难负样本相似度 |
| Gap = S1 - S2 | 越大代表 text3 越"难" |
| validity_ratio | S2 < S1 的比例 |
| method_contribution | 各方法 Gap 均值 |

Stage 1（99 samples）基准结果：Gap 均值 **0.712**，有效率 **97.98%**。
