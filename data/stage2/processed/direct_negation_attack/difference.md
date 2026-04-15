# direct_negation_attack — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 99.5% | 61.7% |
| 平均特征数 | 0.94 | 1.00 |
| 处理时间 (s) | 152.66 | 2353.53 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 5 | 383 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: The dark-skinned man had his knife against the man's bare throat as he was pushing against his back.
- **Regular**: 成功，added: not
- **LLM**: 失败，原因: no_feature_found

- **输入**: A few found purchase on the hull, but most of them were swept away by the train's sheer velocity.
- **Regular**: 成功，added: not
- **LLM**: 失败，原因: no_feature_found

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子