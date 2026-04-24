# role_swap — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 66.4% | 91.4% |
| 平均特征数 | 0.74 | 0.00 |
| 处理时间 (s) | 151.01 | 779.89 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 336 | 86 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: If that happens, you have to mail it back to the store when you get home.
- **Regular**: 成功，[reordered]
- **LLM**: 失败，原因: no_feature_found

- **输入**: The little girl seems to be having fun.
- **Regular**: 成功，[reordered]
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: There is usually no shooting in a home invasion.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，[reordered]

- **输入**: A young person eats.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，a → the

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子