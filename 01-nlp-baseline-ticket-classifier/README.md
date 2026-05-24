# Multilingual Support Ticket Classifier

## Overview

This project is my first completed NLP baseline project in a broader AI / ML engineering portfolio.
The goal was to build a simple but complete text classification workflow for multilingual support tickets before moving to more advanced deep learning and transformer-based solutions.

I treated this project as a learning-focused baseline: understand the data, prepare a clean training set, compare a few classical NLP approaches, save the model, and make it usable outside the notebook.

---

## Business Problem

Support teams receive large volumes of tickets across multiple departments and languages.
The objective of this project is to classify incoming tickets into the correct operational queue so the routing process can be partially automated.

Target classes used in the final baseline:

- Billing and Payments
- Customer Service
- General Inquiry
- Human Resources
- IT Support
- Product Support
- Returns and Exchanges
- Sales and Pre-Sales
- Service Outages and Maintenance
- Technical Support

---

## Dataset

The project uses a multilingual support ticket dataset containing fields such as:

- subject
- body
- queue
- priority
- language
- type
- business_type
- tags

Languages present in the dataset:

- English
- German
- Spanish
- French
- Portuguese

During preprocessing I combined `subject` and `body` into one feature called `context_problem`, because this gave the model a more complete view of the issue described by the customer.

---

## Project Goal

The goal was not to build the best possible production model yet.
The goal was to build a solid classical NLP baseline that can later be compared against PyTorch and transformer-based approaches.

Main learning objectives:

- basic data cleaning for multilingual text
- feature preparation for text classification
- train/test split and baseline evaluation
- comparison of a few TF-IDF based experiments
- model serialization with `joblib`
- simple offline inference

---

## What I Built

This project includes:

- raw and processed data folders
- EDA and preprocessing notebooks
- modular Python files in `src/`
- preprocessing utilities and tests
- training and evaluation scripts
- saved baseline model
- evaluation report and confusion matrix
- simple inference module

At this stage I consider the baseline project complete as a first portfolio milestone.

---

## Workflow

### 1. Data understanding

I first explored the dataset structure, label distribution, language distribution and text lengths.
This helped confirm that the dataset is imbalanced and that some classes are naturally harder to separate.

### 2. Data cleaning and feature preparation

I cleaned noisy records, merged selected business type variants, consolidated tags and created `context_problem` from `subject + body`.

Then I prepared additional text versions:

- `cleaned_context`
- `stemmed_context`

This allowed me to test whether a lighter or stronger normalization step improves baseline performance.

### 3. Train / test split

I created processed training and test datasets and kept the project reproducible through a fixed random state.

### 4. Baseline experiments

I compared several classical NLP setups based on:

- TF-IDF vectorization
- different n-gram ranges
- stopword removal
- Logistic Regression
- LinearSVC

#### Experimental Comparison Summary

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


#### Strategic Project Decision & Data Architecture Analysis

During experimental stage, an additional experiment (ID 9) was conducted, involving the merging of the `Product Support` and `IT Support` classes into a single shared category (reducing the number of classes from 10 to 9). The model performance immediately increased to **61.38% Accuracy** and **58.44% Macro F1-Score**.

Removing this internal label unlocked the potential of trigrams for the remaining classes (for example, the *Sales* department achieved an F1-score increase of 16 percentage points).

#### Final Baseline Selection Decision:
Despite the higher performance of the 9-class variant, **the official production baseline model for this project remains the original 10-class variant (Model ID 8 / ID 2).**

**Why?**
1. **More Challenging Reference Point (Strict Baseline):**  
   The goal of this project is to build a strong foundation for future learning stages. Preserving the full 10-class structure enables a fair and direct comparison between classical Machine Learning approaches and deep learning architectures (Transformers) in future projects.

2. **Business Realism:**  
   In real production environments, reducing the number of classes is not always operationally feasible. The model must learn to handle difficult decision boundaries.

3. **9-Class Variant as a Data Insight:**  
   The 9-class result is documented as a valuable business feedback insight (*Data-Driven Business Insight*), demonstrating how an internal process change within a company can dramatically improve AI automation effectiveness.


### 5. Final baseline selection

For the official baseline, I kept the harder 10-class version instead of simplifying the label space.
I wanted the project to remain a stronger reference point for future improvements.

---

## Final Baseline Result

Selected baseline metrics:

- Accuracy: `0.5737`
- Macro F1: `0.5222`
- Weighted F1: `0.5651`

This result is not meant to be impressive in isolation.
It is meant to be a realistic first benchmark for a multilingual and imbalanced classification problem.

---

## Experiment Summary

The best results came from a `TF-IDF + LinearSVC` pipeline.
Compared with Logistic Regression, LinearSVC handled the sparse text representation more effectively on this dataset.

One additional experiment reduced the problem from 10 classes to 9 by merging `IT Support` and `Product Support`.
That variant achieved stronger metrics, but I kept the original 10-class setup as the official baseline because it is more challenging and more useful for future comparison.

---

## Main Observations

- The dataset is imbalanced, so macro F1 is more informative than accuracy alone.
- `Technical Support` is the largest class and strongly affects the weighted metrics.
- Smaller classes such as `General Inquiry` and `Human Resources` are harder to classify reliably.
- Label overlap between `IT Support` and `Product Support` creates a real ambiguity in the task.
- Classical NLP methods provide a useful benchmark before moving to neural models.

---

## Project Structure

```text
01-nlp-baseline-ticket-classifier/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
└── tests/
```

### How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation from Python or notebook workflow.

Example inference usage:

```python
from src.inference import predict_ticket

prediction = predict_ticket("My laptop cannot connect to the VPN", "en")
print(prediction)
```

The saved baseline model was trained on the `stemmed_context` representation.
The inference module now applies the same basic preprocessing flow used during training: text cleaning followed by language-based stemming when the language is available.

I intentionally kept the inference module simple.
If the language mapping file is unavailable, the code falls back to an empty language dictionary and still allows prediction, which I document as a known baseline limitation rather than overengineering the first version.

---

## Limitations & potential next steps

- the project is a baseline, not a production service
- inference is simple and ready to cover also edge cases
- no FastAPI layer
- no Docker setup
- no experiment tracking 
- add a cleaner CLI or API entry point