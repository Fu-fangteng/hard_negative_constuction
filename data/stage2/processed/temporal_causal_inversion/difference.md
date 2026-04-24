# temporal_causal_inversion — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 1.9% | 58.7% |
| 平均特征数 | 0.02 | 0.00 |
| 处理时间 (s) | 150.74 | 869.63 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 981 | 413 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### LLM 更优
- **输入**: There is usually no shooting in a home invasion.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，added: after, the

- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，sharp → blunt

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子