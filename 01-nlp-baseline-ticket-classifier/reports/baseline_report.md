# Baseline Model Evaluation Report

## Global Metrics
*   **Accuracy:** 0.5737
*   **Macro F1-Score:** 0.5222  
*   **Weighted F1-Score:** 0.5651

## Classification Report per Class
```text
                                 precision    recall  f1-score   support

           Billing and Payments       0.84      0.94      0.89        68
               Customer Service       0.46      0.47      0.46       125
                General Inquiry       0.20      0.09      0.12        11
                Human Resources       0.80      0.36      0.50        11
                     IT Support       0.35      0.28      0.31        89
                Product Support       0.54      0.51      0.52       138
          Returns and Exchanges       0.70      0.77      0.73        39
            Sales and Pre-Sales       0.58      0.41      0.48        27
Service Outages and Maintenance       0.58      0.54      0.56        28
              Technical Support       0.61      0.68      0.64       264

                       accuracy                           0.57       800
                      macro avg       0.57      0.51      0.52       800
                   weighted avg       0.56      0.57      0.57       800



## Visualizations
*   Confusion Matrix was saved as `confusion_matrix.png`.
