# direct_negation_attack — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 90.2% | 61.5% |
| 平均特征数 | 0.96 | 1.00 |
| 处理时间 (s) | 82741.56 | 7.3 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 26935 | 105830 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: You lose the things to the following level if the people recall.
- **Regular**: 成功，added: do, not
- **LLM**: 失败，原因: no_feature_found

- **输入**: The tennis shoes can be in the hundred dollar range.
- **Regular**: 成功，added: not
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: Problems in data synthesis.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，added: are, not, synthesized

- **输入**: I am certainly in agreement with it.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，added: not

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子