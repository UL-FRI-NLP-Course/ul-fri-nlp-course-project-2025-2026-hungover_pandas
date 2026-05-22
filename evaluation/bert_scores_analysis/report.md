# BERTScore Results Analysis

Analyzed 70 CSV files.
Bootstrap settings: 10,000 resamples, 95% percentile confidence intervals, seed 42.

## Overall ranking across scorer models
| evaluated_model | scorer_models | avg_mean_P | avg_mean_R | avg_mean_F1 | sd_across_scorers_F1 | best_scorer_F1 | worst_scorer_F1 | avg_bootstrap_ci95_low_F1 | avg_bootstrap_ci95_high_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma tuned | 14 | 0.7151 | 0.7396 | 0.7240 | 0.0791 | 0.8929 | 0.5929 | 0.7076 | 0.7404 |
| Llama tuned | 14 | 0.7033 | 0.7367 | 0.7166 | 0.0824 | 0.8892 | 0.5794 | 0.7017 | 0.7317 |
| Llama not tuned | 14 | 0.7013 | 0.7176 | 0.7063 | 0.0855 | 0.8861 | 0.5637 | 0.6910 | 0.7219 |
| Gemma not tuned | 14 | 0.7110 | 0.6997 | 0.7025 | 0.0863 | 0.8853 | 0.5564 | 0.6859 | 0.7196 |
| Gemini baseline | 14 | 0.5467 | 0.7214 | 0.6190 | 0.1074 | 0.8401 | 0.4781 | 0.6133 | 0.6248 |

## Tuned vs not tuned mean F1 by scorer
| scorer_model | Gemma not tuned | Gemma tuned | Gemma tuned_minus_not_tuned | Llama not tuned | Llama tuned | Llama tuned_minus_not_tuned | Gemini baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| google_electra-large-discriminator | 0.6241 | 0.6525 | 0.0284 | 0.6312 | 0.6440 | 0.0127 | 0.4995 |
| microsoft_deberta-base-mnli | 0.6868 | 0.7077 | 0.0209 | 0.6885 | 0.6988 | 0.0102 | 0.5989 |
| microsoft_deberta-large-mnli | 0.6403 | 0.6636 | 0.0232 | 0.6440 | 0.6546 | 0.0105 | 0.5217 |
| microsoft_deberta-v2-xlarge | 0.7793 | 0.7949 | 0.0157 | 0.7852 | 0.7934 | 0.0082 | 0.7258 |
| microsoft_deberta-v2-xlarge-mnli | 0.7778 | 0.7922 | 0.0145 | 0.7823 | 0.7901 | 0.0078 | 0.7126 |
| microsoft_deberta-v2-xxlarge | 0.7388 | 0.7559 | 0.0171 | 0.7453 | 0.7542 | 0.0089 | 0.6749 |
| microsoft_deberta-v2-xxlarge-mnli | 0.7582 | 0.7726 | 0.0145 | 0.7627 | 0.7708 | 0.0080 | 0.6920 |
| microsoft_deberta-v3-base | 0.6114 | 0.6414 | 0.0300 | 0.6141 | 0.6294 | 0.0154 | 0.5069 |
| microsoft_deberta-v3-large | 0.6739 | 0.7009 | 0.0271 | 0.6836 | 0.6976 | 0.0141 | 0.6028 |
| microsoft_deberta-v3-small | 0.7683 | 0.7867 | 0.0184 | 0.7679 | 0.7757 | 0.0078 | 0.7015 |
| microsoft_deberta-v3-xsmall | 0.6660 | 0.6921 | 0.0261 | 0.6663 | 0.6762 | 0.0099 | 0.5730 |
| microsoft_deberta-xlarge-mnli | 0.6690 | 0.6891 | 0.0201 | 0.6677 | 0.6795 | 0.0118 | 0.5376 |
| roberta-large | 0.8853 | 0.8929 | 0.0076 | 0.8861 | 0.8892 | 0.0031 | 0.8401 |
| xlnet-large-cased | 0.5564 | 0.5929 | 0.0366 | 0.5637 | 0.5794 | 0.0157 | 0.4781 |

## Scorer model sensitivity
| scorer_model | evaluated_models | avg_F1_across_evaluated_models | spread_between_evaluated_models | mean_std_within_files |
| --- | --- | --- | --- | --- |
| roberta-large | 5 | 0.8787 | 0.0528 | 0.0410 |
| microsoft_deberta-v2-xlarge | 5 | 0.7757 | 0.0691 | 0.0769 |
| microsoft_deberta-v2-xlarge-mnli | 5 | 0.7710 | 0.0796 | 0.0784 |
| microsoft_deberta-v3-small | 5 | 0.7600 | 0.0852 | 0.0815 |
| microsoft_deberta-v2-xxlarge-mnli | 5 | 0.7513 | 0.0806 | 0.0847 |
| microsoft_deberta-v2-xxlarge | 5 | 0.7338 | 0.0810 | 0.0886 |
| microsoft_deberta-base-mnli | 5 | 0.6762 | 0.1088 | 0.1074 |
| microsoft_deberta-v3-large | 5 | 0.6717 | 0.0982 | 0.1206 |
| microsoft_deberta-v3-xsmall | 5 | 0.6547 | 0.1191 | 0.1076 |
| microsoft_deberta-xlarge-mnli | 5 | 0.6486 | 0.1515 | 0.1162 |
| microsoft_deberta-large-mnli | 5 | 0.6248 | 0.1419 | 0.1243 |
| google_electra-large-discriminator | 5 | 0.6103 | 0.1529 | 0.1377 |
| microsoft_deberta-v3-base | 5 | 0.6006 | 0.1346 | 0.1329 |
| xlnet-large-cased | 5 | 0.5541 | 0.1148 | 0.1426 |

## Bootstrap tuning deltas by scorer
| model_family | scorer_model | n_paired_questions | mean_delta_F1 | bootstrap_mean_delta_F1 | bootstrap_ci95_low_delta_F1 | bootstrap_ci95_high_delta_F1 | ci_excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma | xlnet-large-cased | 210 | 0.0366 | 0.0366 | 0.0200 | 0.0544 | True |
| Gemma | microsoft_deberta-v3-base | 210 | 0.0300 | 0.0300 | 0.0143 | 0.0460 | True |
| Gemma | google_electra-large-discriminator | 210 | 0.0284 | 0.0285 | 0.0114 | 0.0464 | True |
| Gemma | microsoft_deberta-v3-large | 210 | 0.0271 | 0.0270 | 0.0129 | 0.0418 | True |
| Gemma | microsoft_deberta-v3-xsmall | 210 | 0.0261 | 0.0260 | 0.0138 | 0.0389 | True |
| Gemma | microsoft_deberta-large-mnli | 210 | 0.0232 | 0.0233 | 0.0086 | 0.0385 | True |
| Gemma | microsoft_deberta-base-mnli | 210 | 0.0209 | 0.0209 | 0.0085 | 0.0340 | True |
| Gemma | microsoft_deberta-xlarge-mnli | 210 | 0.0201 | 0.0202 | 0.0064 | 0.0343 | True |
| Gemma | microsoft_deberta-v3-small | 210 | 0.0184 | 0.0184 | 0.0094 | 0.0280 | True |
| Gemma | microsoft_deberta-v2-xxlarge | 210 | 0.0171 | 0.0171 | 0.0071 | 0.0276 | True |
| Gemma | microsoft_deberta-v2-xlarge | 210 | 0.0157 | 0.0157 | 0.0068 | 0.0249 | True |
| Gemma | microsoft_deberta-v2-xlarge-mnli | 210 | 0.0145 | 0.0146 | 0.0055 | 0.0240 | True |
| Gemma | microsoft_deberta-v2-xxlarge-mnli | 210 | 0.0145 | 0.0145 | 0.0051 | 0.0244 | True |
| Gemma | roberta-large | 210 | 0.0076 | 0.0076 | 0.0033 | 0.0121 | True |
| Llama | xlnet-large-cased | 210 | 0.0157 | 0.0158 | 0.0022 | 0.0301 | True |
| Llama | microsoft_deberta-v3-base | 210 | 0.0154 | 0.0154 | 0.0018 | 0.0296 | True |
| Llama | microsoft_deberta-v3-large | 210 | 0.0141 | 0.0141 | 0.0022 | 0.0267 | True |
| Llama | google_electra-large-discriminator | 210 | 0.0127 | 0.0127 | -0.0007 | 0.0266 | False |
| Llama | microsoft_deberta-xlarge-mnli | 210 | 0.0118 | 0.0119 | -0.0004 | 0.0247 | False |
| Llama | microsoft_deberta-large-mnli | 210 | 0.0105 | 0.0106 | -0.0022 | 0.0239 | False |
| Llama | microsoft_deberta-base-mnli | 210 | 0.0102 | 0.0103 | -0.0009 | 0.0221 | False |
| Llama | microsoft_deberta-v3-xsmall | 210 | 0.0099 | 0.0099 | -0.0011 | 0.0213 | False |
| Llama | microsoft_deberta-v2-xxlarge | 210 | 0.0089 | 0.0089 | 0.0005 | 0.0173 | True |
| Llama | microsoft_deberta-v2-xlarge | 210 | 0.0082 | 0.0082 | 0.0008 | 0.0160 | True |
| Llama | microsoft_deberta-v2-xxlarge-mnli | 210 | 0.0080 | 0.0080 | -0.0003 | 0.0166 | False |
| Llama | microsoft_deberta-v2-xlarge-mnli | 210 | 0.0078 | 0.0078 | -0.0001 | 0.0161 | False |
| Llama | microsoft_deberta-v3-small | 210 | 0.0078 | 0.0078 | -0.0003 | 0.0165 | False |
| Llama | roberta-large | 210 | 0.0031 | 0.0031 | -0.0006 | 0.0070 | False |

## Key observations
- Best average evaluated model across scorer models: **Gemma tuned** with avg mean F1 **0.7240**.
- Gemma tuning delta averaged over scorers: **+0.0214 F1**.
- Gemma bootstrap CIs exclude zero for **14/14** scorer models.
- Llama tuning delta averaged over scorers: **+0.0103 F1**.
- Llama bootstrap CIs exclude zero for **5/14** scorer models.
- Highest absolute-F1 scorer: **roberta-large** with mean F1 **0.8787**.
- Lowest absolute-F1 scorer: **xlnet-large-cased** with mean F1 **0.5541**.