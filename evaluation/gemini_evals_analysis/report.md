# Gemini-as-Judge Evaluation Report

Evaluated **5 models** using Gemini as an automated judge.
Metric focus: `gemini_assessment` (CORRECT / PARTIALLY_CORRECT / INCORRECT / HALLUCINATED) and `gemini_score` (0–100).  Token-overlap F1 is excluded from this analysis.

---

## 1. Overall model ranking

| model | n_valid | mean_score | median_score | std_score | pct_CORRECT | pct_PARTIALLY_CORRECT | pct_INCORRECT | pct_HALLUCINATED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini baseline | 210 | 74.2 | 95.0 | 34.8 | 55.2 | 17.1 | 15.7 | 11.9 |
| Llama tuned | 210 | 70.1 | 80.0 | 31.2 | 39.5 | 32.4 | 22.4 | 5.7 |
| Gemma tuned | 210 | 66.8 | 75.0 | 30.7 | 32.9 | 37.6 | 25.7 | 3.8 |
| Llama not tuned | 114 | 66.3 | 75.0 | 30.9 | 32.5 | 38.6 | 25.4 | 3.5 |
| Gemma not tuned | 210 | 61.0 | 65.0 | 32.8 | 29.5 | 31.0 | 33.8 | 5.7 |

---

## 2. Assessment distribution (%)

| model | CORRECT | PARTIALLY_CORRECT | INCORRECT | HALLUCINATED |
| --- | --- | --- | --- | --- |
| Gemini baseline | 55.2 | 17.1 | 15.7 | 11.9 |
| Llama tuned | 39.5 | 32.4 | 22.4 | 5.7 |
| Gemma tuned | 32.9 | 37.6 | 25.7 | 3.8 |
| Llama not tuned | 32.5 | 38.6 | 25.4 | 3.5 |
| Gemma not tuned | 29.5 | 31.0 | 33.8 | 5.7 |

---

## 3. Tuning impact (paired comparison on shared question IDs)

### Gemma
Paired questions: **210**

| Metric | Not tuned | Tuned | Delta |
| --- | --- | --- | --- |
| Mean score | 61.0 | 66.8 | +5.7 |
| CORRECT % | 29.5 | 32.9 | +3.4 |
| PARTIALLY_CORRECT % | 31.0 | 37.6 | +6.6 |
| INCORRECT % | 33.8 | 25.7 | -8.1 |
| HALLUCINATED % | 5.7 | 3.8 | -1.9 |

### Llama
Paired questions: **114**

| Metric | Not tuned | Tuned | Delta |
| --- | --- | --- | --- |
| Mean score | 66.3 | 74.0 | +7.7 |
| CORRECT % | 32.5 | 43.9 | +11.4 |
| PARTIALLY_CORRECT % | 38.6 | 33.3 | -5.3 |
| INCORRECT % | 25.4 | 18.4 | -7.0 |
| HALLUCINATED % | 3.5 | 4.4 | +0.9 |

---

## 4. Hardest questions (bottom 20 by avg Gemini score across all models)

| id | question | avg_score | n_models | n_correct | n_partial | n_incorrect | n_hallucinated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q110 | I want to find an Erasmus+ exchange at a university focused on data science, analytics, or big data. Which institutions should I check? | 15.0 | 5 | 0 | 0 | 2 | 3 |
| Q157 | Can my traineeship motivation letter be in Slovenian? | 15.0 | 4 | 0 | 0 | 3 | 1 |
| Q117 | Which dormitory is the best for Erasmus students in Prague? | 23.8 | 4 | 0 | 0 | 2 | 2 |
| Q104 | Are there Erasmus+ exchange opportunities in Paris? | 24.0 | 5 | 0 | 0 | 4 | 1 |
| Q35 | What is the monthly Erasmus+ grant for Group 2 countries in 2026/27? | 24.0 | 5 | 0 | 0 | 5 | 0 |
| Q139 | Does University of Padova accept Bachelor students from UL FRI? | 25.0 | 4 | 0 | 1 | 1 | 2 |
| Q100 | Where can I go on Erasmus+ exchange as an undergraduate student? | 25.0 | 5 | 0 | 1 | 3 | 1 |
| Q185 | Can I get Erasmus+ funding if I am writing my thesis abroad and not taking courses? | 31.2 | 4 | 0 | 1 | 2 | 1 |
| Q109 | I want to find an Erasmus+ exchange at a university focused on software engineering and distributed systems. Which institutions should I check? | 32.0 | 5 | 0 | 1 | 2 | 2 |
| Q145 | Which Italian partner university is available for only one Erasmus+ student? | 32.5 | 4 | 0 | 1 | 2 | 1 |
| Q187 | Can I be charged for student union membership? | 32.5 | 4 | 0 | 1 | 2 | 1 |
| Q74 | What must I do after Erasmus+ mobility? | 34.0 | 5 | 1 | 0 | 3 | 1 |
| Q94 | What is the application deadline for Erasmus+ traineeship mobility? | 34.0 | 5 | 0 | 1 | 3 | 1 |
| Q75 | When must I submit documents after returning from Erasmus+? | 35.0 | 5 | 0 | 1 | 4 | 0 |
| Q174 | What should I do if I do not know the exact start and end dates of my exchange? | 35.0 | 4 | 0 | 1 | 2 | 1 |
| Q136 | If I complete 25 ECTS abroad, how many ECTS will be recognized at UL FRI? | 35.0 | 4 | 1 | 0 | 1 | 2 |
| Q25 | What documents do I need for the Erasmus+ financial support application? | 36.0 | 5 | 0 | 1 | 4 | 0 |
| Q97 | Which UL FRI Erasmus+ partner universities are listed in Germany? | 36.0 | 5 | 0 | 2 | 2 | 1 |
| Q66 | Do Erasmus+ students have to pay tuition fees at the host university? | 36.0 | 5 | 1 | 0 | 4 | 0 |
| Q69 | What duties do Erasmus+ students have during mobility? | 36.0 | 5 | 1 | 0 | 3 | 1 |

---

## 5. Key observations

- **Best overall model** by mean Gemini score: **Gemini baseline** with 74.2/100 and 55.2% CORRECT.
- **Weakest model** by mean score: **Gemma not tuned** with 61.0/100.
- **Gemma tuning** improved mean score by +5.7 points (61.0 → 66.8).  CORRECT rate: 29.5% → 32.9% (+3.4 pp).
- **Llama tuning** improved mean score by +7.7 points (66.3 → 74.0).  CORRECT rate: 32.5% → 43.9% (+11.4 pp).
- Highest hallucination rate: **Gemini baseline** (11.9% HALLUCINATED).
- Hardest question across all models: **"I want to find an Erasmus+ exchange at a university focused on data science, ana..."** (avg score 15.0/100, 3 hallucinated out of 5 models).