# Baseline Data Analysis Report: Support Ticket Classification

## 1. Executive Summary

This report provides an initial Exploratory Data Analysis (EDA) of the support ticket dataset. The dataset is multilingual and covers various business sectors. Key preprocessing steps, such as tag consolidation and feature engineering for missing values, have been implemented to ensure data integrity for future modeling.

---

# 2. Dataset Distributions

## 2.1 Ticket Type (Target Variable)

The distribution shows a clear hierarchy in ticket volume. While **"Incident"** is the most frequent category, **"Change"** is significantly underrepresented.

| Type      | Count | Percentage |
|------------|-------|-------------|
| Incident   | 1608  | ~40% |
| Request    | 1097  | ~27.5% |
| Problem    | 853   | ~21.5% |
| Change     | 442   | ~11% |

The distribution shows target variable `Queue` and how many records we have for each Queue of the tickets.

|Queue       | Count |
|------------|--------------------|
|  Technical Support              |   1317 |
|  Product Support                |   690 |
|  Customer Service               |   627 |
|  IT Support                     |   445 |
|  Billing and Payments           |   338 |
|  Returns and Exchanges          |   197 |
|  Service Outages and Maintenance|   141 |
|  Sales and Pre-Sales            |   137 |
|  General Inquiry                |   55 |
|  Human Resources                |   53 |

### Insight
The dataset is imbalanced. Models may struggle to accurately predict the **"Change"** category without specific techniques such as:
- Oversampling
- Weighted loss functions
- Class-balanced sampling

---

## 2.2 Priority Distribution

| Priority | Count |
|----------|-------|
| High     | 1649 |
| Medium   | 1603 |
| Low      | 748 |

### Insight
The majority of tickets are classified as **High** or **Medium** priority. The **Low** priority class represents only about **18%** of the dataset.

---

## 2.3 Language Distribution

The dataset is multilingual and requires a model capable of understanding cross-lingual semantics.

| Language | Count |
|----------|-------|
| EN | 1391 |
| DE | 848 |
| ES | 812 |
| FR | 476 |
| PT | 473 |

---

# 3. Text Length Analysis (String EDA)

Following the concatenation of `subject` and `body` into `context_problem`, the text statistics are as follows:

| Field | Mean (chars) | Median | Max | Min |
|------|---------------|--------|-----|-----|
| Subject | 47.6 | 47.0 | 320.0 | 1.0 |
| Body | 758.5 | 654.0 | 2843.0 | 26.0 |
| Context Problem | 801.4 | 699.0 | 2923.0 | 17.0 |

### Technical Note
A median length of approximately **700 characters** is well suited for Transformer-based architectures such as:
- BERT
- RoBERTa
- XLM-RoBERTa

Most samples should fit within the standard **512-token** limit without significant truncation.

---

# 4. Data Quality & Preprocessing

## 4.1 Missing Value Resolution

### Tags
Consolidating `tag_1` through `tag_8` into a single list effectively handled sparse data across tag columns.

- `tag_9` was identified as entirely empty
- Recommendation: remove `tag_9`

### Subject / Body
By creating the `context_problem` feature (`subject + body`), the dataset successfully resolved:
- 467 missing subject values

This ensures the model always receives the maximum available context.

---

## 4.2 Business Type Anomalies

The `business_type` column contains inconsistent and noisy labels that require normalization.

### Observed Issues

#### Duplicate / Messy Labels
The following labels likely represent the same concept and should be merged:

- `_IT_Services_`
- `Adobe Photoshop 2024`
- `Pit Services`
- `IT Consulting Service`

---

# 5. Sanity Check Observations

A manual review of representative samples (e.g., MacBook battery issues, Jira login problems, software export errors) confirms the following:

## Label Consistency
Ticket priorities and ticket types generally align with the technical severity of the issues described.

## Language Accuracy
The `language` field correctly matches the actual language used in:
- `context_problem`
- `answer`

## Data Integrity
The concatenation strategy successfully preserved contextual meaning, especially for tickets with missing subject values.

---

# 6. Strategic Recommendations

## 6.1 Multilingual Modeling Strategy
Use a pre-trained multilingual Transformer model such as:

- `xlm-roberta-base`

This avoids translation overhead while preserving multilingual semantic relationships.

---

## 6.2 Outlier Handling
Review or remove records where:

context_problem length < 20 characters

Such entries are unlikely to contain enough diagnostic information for reliable classification.

## 6.3 Evaluation Metrics

Use Macro-F1 Score instead of plain Accuracy.

This ensures balanced evaluation performance across minority classes such as:

"Change"
"Low" priority tickets

Accuracy alone would likely overestimate real-world performance due to class imbalance.