# premise_disruption — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 100.0% | 95.7% |
| 平均特征数 | 1.00 | 1.00 |
| 处理时间 (s) | 146.42 | 1203.05 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 0 | 43 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: Page two of the quiz is worth looking at, if you don't already.
- **Regular**: 成功，added: even
- **LLM**: 失败，原因: no_feature_found

- **输入**: One visit happened in April.
- **Regular**: 成功，added: theory
- **LLM**: 失败，原因: no_feature_found

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子