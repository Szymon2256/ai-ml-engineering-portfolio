# Pytorch Model Report
## Assumption & Decisions
- source of data: `ticket_data` from project 01
- input variable: `cleaned_context`
- target to predict: `queue`
- number of labels in `queue`: 10
    - we froze number of labels and use 10 labels of `queue` as it was in project 01
    - labels which we will use: ['Billing and Payments', 'Customer Service', 'General Inquiry', 'Human Resources', 'IT Support', 'Product Support', 'Returns and Exchanges', 'Sales and Pre-Sales', 'Service Outages and Maintenance', 'Technical Support']
- first tokenizer - whitespace tokenizer, we will use `text.split()`
- established `max sequence length` for tokenization - 256 tokens:
    This decision balances computational efficiency with data preservation based on the dataset's length distribution
- Special tokens: `PAD=0`, `UNK=1` for out-of-vocabulary words. BOS/EOS not used — classification task does not require sequence boundary markers.
- `vocab_size=10,000` was selected as the point where coverage is high (98.21%), whole training set contains 16391 unique tokens, additional 5000 tokens in vocab size doesn't extend our vocabulary too much (only 1.41% more) but it extend embedding matrix size by 50%, not worth to extend it
- our training data is imbalanced (as it was in project 01), the distribution of the data is follow:
Technical Support                  1053
Product Support                     552
Customer Service                    502
IT Support                          356
Billing and Payments                270
Returns and Exchanges               158
Service Outages and Maintenance     113
Sales and Pre-Sales                 110
General Inquiry                      44
Human Resources                      42
We need to keep it in mind when creating validation set, also we will need to use class weight balance methods to handle this problem
- test split - same as in 01
- validation split: created from train set in 02_eda_processing, we take 15% of train data as a validation set, we also stratify the data by `queue` feature
