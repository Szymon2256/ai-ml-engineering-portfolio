# Pytorch Approach to Ticket Classification

## Scope
- same as in project 01
- input variable: `cleaned_context`
- target to predict: `queue`
- comparison with results from project 01

## Assumption
- source of data: `ticket_data` from project 01
- input variable: `cleaned_context`
- target to predict: `queue`
- number of labels in `queue`: 10
    - we froze number of labels and use 10 labels of `queue` as it was in project 01
    - labels which we will use: ['Billing and Payments', 'Customer Service', 'General Inquiry', 'Human Resources', 'IT Support', 'Product Support', 'Returns and Exchanges', 'Sales and Pre-Sales', 'Service Outages and Maintenance', 'Technical Support']
- test split - same as in 01
- validation split: created from train set later

## Requirements
- end-to-end training for 1st Pytorch model
- checkpoint saving (models)
- metrics of the model:
    - accuracy 
    - macro F1 
    - weighted F1 
    - confusion matrix 
    - comparison with Project 01 baseline 
- end-to-end training for 2nd Pytorch model
- model comparison
