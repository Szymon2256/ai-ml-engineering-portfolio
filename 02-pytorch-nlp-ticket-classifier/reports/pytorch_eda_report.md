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


### First results after first run without balancing the data
Using device: cpu
Loading data...
2026-08-14 21:52:01,748 - INFO - Splitting data into train and test sets with test size 0.15 and random state 42...
2026-08-14 21:52:01,753 - INFO - Data splitting completed.
2026-08-14 21:52:01,753 - INFO - Train:      2,720
2026-08-14 21:52:01,753 - INFO - Validation: 480
2026-08-14 21:52:01,753 - INFO - Test:       800
2026-08-14 21:52:01,875 - INFO - Vocabulary size: 10,000
2026-08-14 21:52:01,875 - INFO - Classes (10): ['Billing and Payments', 'Customer Service', 'General Inquiry', 'Human Resources', 'IT Support', 'Product Support', 'Returns and Exchanges', 'Sales and Pre-Sales', 'Service Outages and Maintenance', 'Technical Support']
Train batches:      85
Validation batches: 15
Test batches:       25

Model parameters: 1,281,290

Training for 30 epochs...

Epoch 01/30 | train_loss: 2.1442 | val_loss: 1.9956 | val_acc: 0.3250 | val_macro_f1: 0.0491 | val_weighted_f1: 0.1617
 --> Checkpoint saved at epoch 1
Epoch 02/30 | train_loss: 1.9096 | val_loss: 1.8683 | val_acc: 0.3292 | val_macro_f1: 0.0518 | val_weighted_f1: 0.1666
 --> Checkpoint saved at epoch 2
Epoch 03/30 | train_loss: 1.8116 | val_loss: 1.8151 | val_acc: 0.3333 | val_macro_f1: 0.0569 | val_weighted_f1: 0.1759
 --> Checkpoint saved at epoch 3
Epoch 04/30 | train_loss: 1.7424 | val_loss: 1.7633 | val_acc: 0.3354 | val_macro_f1: 0.0740 | val_weighted_f1: 0.1981
 --> Checkpoint saved at epoch 4
Epoch 05/30 | train_loss: 1.6678 | val_loss: 1.7080 | val_acc: 0.3625 | val_macro_f1: 0.1091 | val_weighted_f1: 0.2503
 --> Checkpoint saved at epoch 5
Epoch 06/30 | train_loss: 1.5870 | val_loss: 1.6493 | val_acc: 0.4083 | val_macro_f1: 0.1624 | val_weighted_f1: 0.3163
 --> Checkpoint saved at epoch 6
Epoch 07/30 | train_loss: 1.5043 | val_loss: 1.5957 | val_acc: 0.4229 | val_macro_f1: 0.1869 | val_weighted_f1: 0.3430
 --> Checkpoint saved at epoch 7
Epoch 08/30 | train_loss: 1.4213 | val_loss: 1.5428 | val_acc: 0.4437 | val_macro_f1: 0.2191 | val_weighted_f1: 0.3707
 --> Checkpoint saved at epoch 8
Epoch 09/30 | train_loss: 1.3410 | val_loss: 1.4973 | val_acc: 0.4667 | val_macro_f1: 0.2544 | val_weighted_f1: 0.4022
 --> Checkpoint saved at epoch 9
Epoch 10/30 | train_loss: 1.2631 | val_loss: 1.4554 | val_acc: 0.4792 | val_macro_f1: 0.2693 | val_weighted_f1: 0.4166
 --> Checkpoint saved at epoch 10
Epoch 11/30 | train_loss: 1.1897 | val_loss: 1.4181 | val_acc: 0.5021 | val_macro_f1: 0.2924 | val_weighted_f1: 0.4413
 --> Checkpoint saved at epoch 11
Epoch 12/30 | train_loss: 1.1210 | val_loss: 1.3868 | val_acc: 0.5229 | val_macro_f1: 0.3221 | val_weighted_f1: 0.4702
 --> Checkpoint saved at epoch 12
Epoch 13/30 | train_loss: 1.0545 | val_loss: 1.3632 | val_acc: 0.5375 | val_macro_f1: 0.3362 | val_weighted_f1: 0.4855
 --> Checkpoint saved at epoch 13
Epoch 14/30 | train_loss: 0.9911 | val_loss: 1.3395 | val_acc: 0.5500 | val_macro_f1: 0.3609 | val_weighted_f1: 0.5037
 --> Checkpoint saved at epoch 14
Epoch 15/30 | train_loss: 0.9314 | val_loss: 1.3217 | val_acc: 0.5458 | val_macro_f1: 0.3582 | val_weighted_f1: 0.5024
Epoch 16/30 | train_loss: 0.8744 | val_loss: 1.3070 | val_acc: 0.5542 | val_macro_f1: 0.3818 | val_weighted_f1: 0.5136
 --> Checkpoint saved at epoch 16
Epoch 17/30 | train_loss: 0.8198 | val_loss: 1.2946 | val_acc: 0.5625 | val_macro_f1: 0.4015 | val_weighted_f1: 0.5262
 --> Checkpoint saved at epoch 17
Epoch 18/30 | train_loss: 0.7665 | val_loss: 1.2824 | val_acc: 0.5646 | val_macro_f1: 0.4104 | val_weighted_f1: 0.5303
 --> Checkpoint saved at epoch 18
Epoch 19/30 | train_loss: 0.7179 | val_loss: 1.2739 | val_acc: 0.5625 | val_macro_f1: 0.4113 | val_weighted_f1: 0.5292
 --> Checkpoint saved at epoch 19
Epoch 20/30 | train_loss: 0.6693 | val_loss: 1.2684 | val_acc: 0.5521 | val_macro_f1: 0.4035 | val_weighted_f1: 0.5211
Epoch 21/30 | train_loss: 0.6239 | val_loss: 1.2638 | val_acc: 0.5563 | val_macro_f1: 0.4063 | val_weighted_f1: 0.5252
Epoch 22/30 | train_loss: 0.5806 | val_loss: 1.2609 | val_acc: 0.5563 | val_macro_f1: 0.4086 | val_weighted_f1: 0.5264
Epoch 23/30 | train_loss: 0.5401 | val_loss: 1.2588 | val_acc: 0.5563 | val_macro_f1: 0.4047 | val_weighted_f1: 0.5253
Epoch 24/30 | train_loss: 0.5018 | val_loss: 1.2584 | val_acc: 0.5667 | val_macro_f1: 0.4192 | val_weighted_f1: 0.5379
 --> Checkpoint saved at epoch 24
Epoch 25/30 | train_loss: 0.4652 | val_loss: 1.2576 | val_acc: 0.5667 | val_macro_f1: 0.4201 | val_weighted_f1: 0.5404
 --> Checkpoint saved at epoch 25
Epoch 26/30 | train_loss: 0.4309 | val_loss: 1.2629 | val_acc: 0.5750 | val_macro_f1: 0.4324 | val_weighted_f1: 0.5500
 --> Checkpoint saved at epoch 26
Epoch 27/30 | train_loss: 0.3995 | val_loss: 1.2625 | val_acc: 0.5813 | val_macro_f1: 0.4416 | val_weighted_f1: 0.5568
 --> Checkpoint saved at epoch 27
Epoch 28/30 | train_loss: 0.3689 | val_loss: 1.2652 | val_acc: 0.5833 | val_macro_f1: 0.4487 | val_weighted_f1: 0.5594
 --> Checkpoint saved at epoch 28
Epoch 29/30 | train_loss: 0.3408 | val_loss: 1.2705 | val_acc: 0.5833 | val_macro_f1: 0.4383 | val_weighted_f1: 0.5590
Epoch 30/30 | train_loss: 0.3148 | val_loss: 1.2747 | val_acc: 0.5875 | val_macro_f1: 0.4789 | val_weighted_f1: 0.5660
 --> Checkpoint saved at epoch 30

Loading best checkpoint...

=== Test Results ===
Test accuracy:    0.5312
Test macro F1:    0.4430
Test weighted F1: 0.5163
Best epoch:       30

### Second run
Using device: cpu
Loading data...
2026-08-14 22:06:30,728 - INFO - Splitting data into train and test sets with test size 0.15 and random state 42...
2026-08-14 22:06:30,731 - INFO - Data splitting completed.
2026-08-14 22:06:30,733 - INFO - Train:      2,720
2026-08-14 22:06:30,733 - INFO - Validation: 480
2026-08-14 22:06:30,733 - INFO - Test:       800
2026-08-14 22:06:30,857 - INFO - Vocabulary size: 10,000
2026-08-14 22:06:30,857 - INFO - Classes (10): ['Billing and Payments', 'Customer Service', 'General Inquiry', 'Human Resources', 'IT Support', 'Product Support', 'Returns and Exchanges', 'Sales and Pre-Sales', 'Service Outages and Maintenance', 'Technical Support']
Train batches:      85
Validation batches: 15
Test batches:       25

Model parameters: 1,281,290

Class weights: [np.float64(1.185), np.float64(0.637), np.float64(7.273), np.float64(7.619), np.float64(0.899), np.float64(0.58), np.float64(2.025), np.float64(2.909), np.float64(2.832), np.float64(0.304)]

Training for 30 epochs...

Epoch 01/30 | train_loss: 2.2965 | val_loss: 2.2818 | val_acc: 0.1625 | val_macro_f1: 0.1129 | val_weighted_f1: 0.1690
 --> Checkpoint saved at epoch 1
Epoch 02/30 | train_loss: 2.2407 | val_loss: 2.2451 | val_acc: 0.2458 | val_macro_f1: 0.2056 | val_weighted_f1: 0.2394
 --> Checkpoint saved at epoch 2
Epoch 03/30 | train_loss: 2.1785 | val_loss: 2.2019 | val_acc: 0.2979 | val_macro_f1: 0.2431 | val_weighted_f1: 0.2855
 --> Checkpoint saved at epoch 3
Epoch 04/30 | train_loss: 2.1018 | val_loss: 2.1462 | val_acc: 0.3146 | val_macro_f1: 0.2511 | val_weighted_f1: 0.2982
 --> Checkpoint saved at epoch 4
Epoch 05/30 | train_loss: 2.0059 | val_loss: 2.0821 | val_acc: 0.3229 | val_macro_f1: 0.2663 | val_weighted_f1: 0.3024
 --> Checkpoint saved at epoch 5
Epoch 06/30 | train_loss: 1.8944 | val_loss: 2.0078 | val_acc: 0.3521 | val_macro_f1: 0.2988 | val_weighted_f1: 0.3317
 --> Checkpoint saved at epoch 6
Epoch 07/30 | train_loss: 1.7763 | val_loss: 1.9399 | val_acc: 0.3521 | val_macro_f1: 0.3028 | val_weighted_f1: 0.3239
 --> Checkpoint saved at epoch 7
Epoch 08/30 | train_loss: 1.6579 | val_loss: 1.8744 | val_acc: 0.3604 | val_macro_f1: 0.3125 | val_weighted_f1: 0.3348
 --> Checkpoint saved at epoch 8
Epoch 09/30 | train_loss: 1.5309 | val_loss: 1.8186 | val_acc: 0.3750 | val_macro_f1: 0.3209 | val_weighted_f1: 0.3552
 --> Checkpoint saved at epoch 9
Epoch 10/30 | train_loss: 1.4120 | val_loss: 1.7639 | val_acc: 0.4021 | val_macro_f1: 0.3412 | val_weighted_f1: 0.3871
 --> Checkpoint saved at epoch 10
Epoch 11/30 | train_loss: 1.3041 | val_loss: 1.7173 | val_acc: 0.4104 | val_macro_f1: 0.3641 | val_weighted_f1: 0.3958
 --> Checkpoint saved at epoch 11
Epoch 12/30 | train_loss: 1.2014 | val_loss: 1.6797 | val_acc: 0.4208 | val_macro_f1: 0.3785 | val_weighted_f1: 0.4091
 --> Checkpoint saved at epoch 12
Epoch 13/30 | train_loss: 1.1028 | val_loss: 1.6446 | val_acc: 0.4354 | val_macro_f1: 0.3908 | val_weighted_f1: 0.4266
 --> Checkpoint saved at epoch 13
Epoch 14/30 | train_loss: 1.0109 | val_loss: 1.6180 | val_acc: 0.4500 | val_macro_f1: 0.4112 | val_weighted_f1: 0.4443
 --> Checkpoint saved at epoch 14
Epoch 15/30 | train_loss: 0.9316 | val_loss: 1.5932 | val_acc: 0.4604 | val_macro_f1: 0.4236 | val_weighted_f1: 0.4560
 --> Checkpoint saved at epoch 15
Epoch 16/30 | train_loss: 0.8601 | val_loss: 1.5790 | val_acc: 0.4625 | val_macro_f1: 0.4267 | val_weighted_f1: 0.4584
 --> Checkpoint saved at epoch 16
Epoch 17/30 | train_loss: 0.7962 | val_loss: 1.5611 | val_acc: 0.4813 | val_macro_f1: 0.4379 | val_weighted_f1: 0.4793
 --> Checkpoint saved at epoch 17
Epoch 18/30 | train_loss: 0.7281 | val_loss: 1.5533 | val_acc: 0.4854 | val_macro_f1: 0.4389 | val_weighted_f1: 0.4849
 --> Checkpoint saved at epoch 18
Epoch 19/30 | train_loss: 0.6749 | val_loss: 1.5492 | val_acc: 0.4917 | val_macro_f1: 0.4430 | val_weighted_f1: 0.4914
 --> Checkpoint saved at epoch 19
Epoch 20/30 | train_loss: 0.6177 | val_loss: 1.5397 | val_acc: 0.4938 | val_macro_f1: 0.4482 | val_weighted_f1: 0.4939
 --> Checkpoint saved at epoch 20
Epoch 21/30 | train_loss: 0.5792 | val_loss: 1.5462 | val_acc: 0.5083 | val_macro_f1: 0.4613 | val_weighted_f1: 0.5070
 --> Checkpoint saved at epoch 21
Epoch 22/30 | train_loss: 0.5324 | val_loss: 1.5415 | val_acc: 0.5167 | val_macro_f1: 0.4747 | val_weighted_f1: 0.5157
 --> Checkpoint saved at epoch 22
Epoch 23/30 | train_loss: 0.4967 | val_loss: 1.5512 | val_acc: 0.5125 | val_macro_f1: 0.4689 | val_weighted_f1: 0.5113
Epoch 24/30 | train_loss: 0.4627 | val_loss: 1.5539 | val_acc: 0.5125 | val_macro_f1: 0.4678 | val_weighted_f1: 0.5133
Epoch 25/30 | train_loss: 0.4261 | val_loss: 1.5684 | val_acc: 0.5125 | val_macro_f1: 0.4577 | val_weighted_f1: 0.5128
Epoch 26/30 | train_loss: 0.3924 | val_loss: 1.5712 | val_acc: 0.5125 | val_macro_f1: 0.4636 | val_weighted_f1: 0.5128
Epoch 27/30 | train_loss: 0.3645 | val_loss: 1.5806 | val_acc: 0.5167 | val_macro_f1: 0.4673 | val_weighted_f1: 0.5152
Epoch 28/30 | train_loss: 0.3403 | val_loss: 1.5929 | val_acc: 0.5146 | val_macro_f1: 0.4647 | val_weighted_f1: 0.5130
Epoch 29/30 | train_loss: 0.3177 | val_loss: 1.6087 | val_acc: 0.5125 | val_macro_f1: 0.4691 | val_weighted_f1: 0.5104
Epoch 30/30 | train_loss: 0.2936 | val_loss: 1.6194 | val_acc: 0.5167 | val_macro_f1: 0.4750 | val_weighted_f1: 0.5146
 --> Checkpoint saved at epoch 30

Loading best checkpoint...

=== Test Results ===
Test accuracy:    0.4988
Test macro F1:    0.4287
Test weighted F1: 0.4926
Best epoch:       30

