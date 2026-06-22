# Pytorch Approach to Ticket Classification

## Scope
- same as in project 01
- input variable: `cleaned_context`
- target to predict: `queue`
- comparison with results from project 01

## Assumption & Decisions
- source of data: `ticket_data` from project 01
- input variable: `cleaned_context`
- target to predict: `queue`
- number of labels in `queue`: 10
- test split - same as in 01
- validation split: created from train set in 02_eda_processing, we take 15% of train data as a validation set, we also stratify the data by `queue` feature

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
