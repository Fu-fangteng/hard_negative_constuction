# entity_pronoun_substitution — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 20.3% | 35.4% |
| 平均特征数 | 0.80 | 0.00 |
| 处理时间 (s) | 157.27 | 802.58 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 797 | 646 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: A few found purchase on the hull, but most of them were swept away by the train's sheer velocity.
- **Regular**: 成功，them → it
- **LLM**: 失败，原因: no_feature_found

- **输入**: The Kal swung his club high.
- **Regular**: 成功，his → her
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，[punctuation only]

- **输入**: We went to see Kindergarten Cop not so long ago.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，kindergarten, cop → the, office

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子