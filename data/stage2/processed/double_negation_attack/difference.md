# double_negation_attack — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 8.1% | 23.1% |
| 平均特征数 | 0.06 | 0.00 |
| 处理时间 (s) | 149.07 | 863.85 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 919 | 769 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: Not all evaluations diverged from the questions that were asked.
- **Regular**: 成功，removed: not
- **LLM**: 失败，原因: no_feature_found

- **输入**: I would, without a doubt, do that, son!
- **Regular**: 成功，removed: without
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，[punctuation only]

- **输入**: A dog is at the beach
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，[punctuation only]

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子