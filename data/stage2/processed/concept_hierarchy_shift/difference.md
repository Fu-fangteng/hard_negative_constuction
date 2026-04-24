# concept_hierarchy_shift — Regular vs LLM 对比报告

## 总体统计

| 指标 | Regular | LLM |
|---|---|---|
| 成功率 | 3.9% | 84.6% |
| 平均特征数 | 0.58 | 0.00 |
| 处理时间 (s) | 151.09 | 913.06 |

## 失败原因对比

| 失败原因 | Regular | LLM |
|---|---|---|
| no_feature_found | 961 | 154 |
| output_same_as_input | 0 | 0 |
| empty_output | 0 | 0 |
| exception | 0 | 0 |

## 典型案例

### Regular 更优
- **输入**: A dark city street containing one woman walking.
- **Regular**: 成功，city → place
- **LLM**: 失败，原因: no_feature_found

- **输入**: If flight capabilities appear in movies, they should be shown consistently and logically.
- **Regular**: 成功，movies → films
- **LLM**: 失败，原因: no_feature_found

### LLM 更优
- **输入**: The dark-skinned man had his knife against the man's bare throat as he was pushing against his back.
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，back → chest

- **输入**: The rocks have sharp angles
- **Regular**: 失败，原因: no_feature_found
- **LLM**: 成功，rocks → features

## 结论

- **Regular** 推荐场景：含明确数值/逻辑词/专有名词的句子
- **LLM** 推荐场景：含复杂实体、代词指代、语义理解需求的句子