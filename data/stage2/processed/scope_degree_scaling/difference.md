# scope_degree_scaling — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 7.5% | 37.0% |
| 平均特征数 | 0.08 | 0.00 |
| 处理时间 (s) | 151.97 | 894.78 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 925 | 630 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: We are all aware that INS is stretched nearly to the limit.
- **Regular**: 成功，all → some
- **LLM**: 失败，原因: no_feature_found

- **输入**: A man is near some religious items.
- **Regular**: 成功，some → most
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: There is usually no shooting in a home invasion.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，usually → rarely

- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，angles → edges

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子