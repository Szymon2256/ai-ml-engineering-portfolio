# Multilingual Support Ticket Classifier

## Business problem

Customer support teams receive tickets in multiple languages. 
The goal is to automatically classify incoming tickets into operational categories to support routing, prioritization and reporting.

---

## Technical approach


I started with classical NLP baselines using TF-IDF features and linear classifiers.
This provides an interpretable, fast and strong benchmark before moving to transformer-based models.

---

## Evaluation

The model is evaluated using macro F1, weighted F1 and confusion matrix.
Macro F1 is important because some ticket categories may be underrepresented.

# First result of prediction

## Global Metrics
*   **Accuracy:** 0.4713
*   **Macro F1-Score:** 0.4761  
*   **Weighted F1-Score:** 0.4683

## Classification Report per Class
```text
                                 precision    recall  f1-score   support

           Billing and Payments       0.79      0.91      0.85        68
               Customer Service       0.41      0.28      0.33       125
                General Inquiry       0.29      0.45      0.36        11
                Human Resources       0.43      0.55      0.48        11
                     IT Support       0.30      0.38      0.34        89
                Product Support       0.40      0.48      0.43       138
          Returns and Exchanges       0.50      0.79      0.61        39
            Sales and Pre-Sales       0.35      0.52      0.42        27
Service Outages and Maintenance       0.33      0.71      0.45        28
              Technical Support       0.64      0.39      0.49       264

                       accuracy                           0.47       800
                      macro avg       0.44      0.55      0.48       800
                   weighted avg       0.50      0.47      0.47       800
```
---
## Experimental Comparison Summary

| ID | TF-IDF Vectorization | Classifier | Accuracy | Macro F1 | Weighted F1 | Key Findings |
|---|---|---|---|---|---|---|
| 1 | Unigram + Bigram (1,2), `max_features=5000` | Logistic Regression | 0.4713 | 0.4761 | 0.4683 | Initial baseline without advanced preprocessing. Weakest overall performance. |
| 2 | Unigram (1,1), stopwords removal, `max_features=5000` | LinearSVC | 0.5587 | **0.5406** | 0.5554 | Best-performing experiment. Achieved the strongest balance across all evaluation metrics. |
| 3 | Unigram + Bigram (1,2), stopwords removal, `max_features=5000` | LinearSVC | 0.5487 | 0.5179 | 0.5448 | Slight performance degradation. Adding bigrams with a limited feature space reduced generalization quality. |
| 4 | Character n-grams (3,5), stopwords removal, `max_features=5000` | LinearSVC | 0.5238 | 0.5096 | 0.5209 | Stable multilingual performance, but character n-grams were constrained by the feature limit. |
| 5 | Unigram + Bigram + Trigram (1,3), stopwords removal, `max_features=5000` | LinearSVC | 0.5475 | 0.4881 | 0.5409 | Worst-performing SVM variant. Excessive feature-space expansion caused strong sparsity effects. |
| 6 | Unigram + Bigram (1,2), stopwords removal, `max_features=5000` | Logistic Regression | 0.4838 | 0.4661 | 0.4838 | Logistic Regression handled sparse high-dimensional matrices less effectively than LinearSVC. |
| 7 | Unigram (1,1), stop_words removal, `max_features=5000` | Logistic Regression | 0.4975 | 0.5020 | 0.4963 | Significant improvement over the bigram Logistic Regression configuration. |
| 8 | Uni+Bi+Tri (1,3), stop_words removal, `max_features=30000` | LinearSVC | **0.5737** | 0.5222 | **0.5651** | Best in terms of accuracy for 10 classess |
| 9 | Uni+Bi+Tri (1,3), stop_words removal, `max_feat=30000` | LinearSVC | **0.6138** | **0.5844** | **0.6071** | 9 classess (IT Support + Product Supprot merged), Highest global metrics. Removing class conflicts unlocked the potential of trigrams for the rest of the dataset. |

---


---

# Context (TBD)
Project overview
Problem statement
Dataset
Approach
Models tested
Metrics
Results
Error analysis
How to run
Next steps

### Checklist

- [ ] Mam repo na GitHubie
- [ ] Mam dataset ticketów
- [ ] Mam EDA
- [ ] Mam preprocessing
- [ ] Mam minimum 3 eksperymenty modelowe
- [ ] Mam metryki: accuracy, macro F1, weighted F1
- [ ] Mam confusion matrix
- [ ] Mam error analysis
- [ ] Mam zapisany model
- [ ] Mam inference.py
- [ ] Mam README
- [ ] Mam opis next steps

