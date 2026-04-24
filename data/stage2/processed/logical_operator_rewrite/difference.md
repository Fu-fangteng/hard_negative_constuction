# logical_operator_rewrite — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 5.2% | 21.2% |
| 平均特征数 | 0.05 | 0.00 |
| 处理时间 (s) | 1073.85 | 1140.88 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 948 | 788 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: If that happens, you have to mail it back to the store when you get home.
- **Regular**: 成功，if → unless
- **LLM**: 失败，原因: no_feature_found

- **输入**: Two people wear white hats while crossing a street.
- **Regular**: 成功，while → when
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，[punctuation only]

- **输入**: The Kal swung his club high.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，high → low

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子