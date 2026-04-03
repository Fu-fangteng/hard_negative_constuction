# Evaluation Report

**Generated:** evaluator

**Samples:** 99

**Similarity Backend:** angle_emb

**GT Score Mapping:** 0–5 → 0–1


---

## 📊 Key Metrics

| Metric | S1 (Text1-Text2) | S2 (Text1-Text3) | Gap (S1 - S2) |
|--------|------------------|------------------|---------------|
| **Mean** | 4.8310 | 4.1191 | 0.7119 |
| **Median** | 4.8926 | 4.1392 | 0.6043 |
| **Variance** | 0.0386 | 0.2512 | 0.3019 |
| **Validity Ratio** (S2 < S1) | - | - | 97.98% |


## 🎯 Ground Truth Offset

| Metric | Value |
|--------|-------|
| Mean Offset | 3.1191 |
| Median Offset | 3.1392 |
| Variance | 0.2512 |
| Correlation (S2 vs GT) | nan |
| Spearman Correlation | nan |


## 🔬 Method Contribution Analysis

| Method | Samples | Mean Gap | Median Gap | Variance |
|--------|---------|----------|------------|----------|
| direct_negation_attack | 59 | 1.0452 | 1.0998 | 0.2083 |
| double_negation_attack | 2 | 0.3806 | 0.3806 | 0.0016 |
| numeric_metric_transform | 20 | 0.3207 | 0.2214 | 0.0245 |
| entity_pronoun_substitution | 13 | 0.1008 | 0.0412 | 0.0227 |
| scope_degree_scaling | 5 | 0.0649 | 0.0331 | 0.0052 |


## 🏆 Top Methods by Mean Gap

| Rank | Method | Mean Gap | Samples |
|------|--------|----------|---------|
| 1 | direct_negation_attack | 1.0452 | 59 |
| 2 | double_negation_attack | 0.3806 | 2 |
| 3 | numeric_metric_transform | 0.3207 | 20 |
| 4 | entity_pronoun_substitution | 0.1008 | 13 |
| 5 | scope_degree_scaling | 0.0649 | 5 |


## 📈 Distribution Summary

### S1 Distribution (Text1-Text2)

- **Range:** [3.9062, 5.0000]
- **Mean ± Std:** 4.8310 ± 0.1965

### S2 Distribution (Text1-Text3)

- **Range:** [3.0577, 4.9378]
- **Mean ± Std:** 4.1191 ± 0.5012

### Gap Distribution

- **Range:** [-0.0046, 1.8525]
- **Mean ± Std:** 0.7119 ± 0.5495
- **Positive Gap Ratio:** 97.98%


### GT Score Distribution (Mapped to 0-1)

- **Range:** [1.0000, 1.0000]
- **Mean ± Std:** 1.0000 ± 0.0000


## 📝 Sample Data (First 10 rows)

| ID | Text1 (truncated) | Text2 (truncated) | Text3 (truncated) | S1 | S2 | Gap | GT (mapped) |
|----|-------------------|-------------------|-------------------|----|----|-----|-------------|
| sample_000010 | he later learned that the incident was caused by t... | he later found out the alarming incident had been ... | she later found out the alarming incident had been... | 4.8480 | 4.5424 | 0.3056 | 1.0000 |
| sample_000024 | aaa spokesman jerry cheske said prices may have af... | aaa spokesman jerry cheske said prices might have ... | aaa spokesman jerry cheske said prices might have ... | 4.8833 | 4.8879 | -0.0046 | 1.0000 |
| sample_000082 | after protesters rushed the stage and twice cut po... | after protesters rushed the stage and twice cut po... | after protesters rushed the stage and twice cut po... | 4.6314 | 4.5634 | 0.0680 | 1.0000 |
| sample_000085 | doctors say one or both boys may die , and that so... | doctors said that one or both of the boys may die ... | doctors said that one or both of the boys may die ... | 4.9453 | 4.9378 | 0.0076 | 1.0000 |
| sample_000102 | the company claims it 's the largest single apple ... | the company claimed it is the largest sale of xser... | the company claimed they is the largest sale of xs... | 4.5802 | 4.5468 | 0.0335 | 1.0000 |
| sample_000107 | the value will total about $ 124 million , includi... | including convertible securities , the total estim... | including convertible securities , the total estim... | 4.8023 | 4.3940 | 0.4082 | 1.0000 |
| sample_000139 | but software license revenues , a measure financia... | license sales , a key measure of demand , fell 21 ... | license sales , a key measure of demand , fell 22 ... | 4.2795 | 3.6771 | 0.6024 | 1.0000 |
| sample_000210 | sendmail said the system can even be set up to per... | the product can be instructed to permit business-o... | the product can not be instructed to permit busine... | 4.5863 | 3.6133 | 0.9730 | 1.0000 |
| sample_000264 | the operating revenues were $ 1.45 billion , an in... | operating revenues rose to $ 1.45 billion from $ 1... | operating revenues rose to $ 2.45 billion from $ 1... | 4.8739 | 4.6618 | 0.2122 | 1.0000 |
| sample_000274 | ruffner , 45 , doesn 't yet have an attorney in th... | ruffner , 45 , does not have a lawyer on the murde... | ruffner , 46 , does not have a lawyer on the murde... | 4.8253 | 4.6822 | 0.1431 | 1.0000 |

