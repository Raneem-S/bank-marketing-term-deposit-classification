# Bank Marketing: Predicting Term Deposit Subscriptions

Predicting which clients will subscribe to a term deposit **before** they are called, so a bank can prioritise its phone campaign. Part of the IBM Machine Learning Professional Certificate (Supervised Learning: Classification).

## Question

Which clients are most likely to subscribe, and how much more efficient would a model-driven call list be than calling at random?

## Data

- **Source:** [Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), UCI Machine Learning Repository (ID 222). Phone campaigns by a Portuguese bank, 2008–2010.
- **Size:** 45,211 clients, 16 features (client profile, current campaign contact, previous campaigns)
- **Target:** `y`, subscribed to a term deposit. Only **11.7%** said yes, so the classes are imbalanced.
- The notebook downloads the data directly with `ucimlrepo`; no data files are stored in this repo.

## Approach

1. Removed `duration` (call length). It is only known after the call, so using it would be **target leakage**.
2. Kept missing categorical values as `"unknown"`, and replaced the `pdays = -1` code with a `previously_contacted` flag.
3. One-hot encoded categories (42 features) and used a **stratified** 80/20 train/test split.
4. Compared four models on precision, recall, F1 and ROC-AUC, because accuracy is misleading at 88% "no".

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.894 | 0.668 | 0.181 | 0.284 | 0.772 | 0.767 |
| Logistic Regression (balanced) | 0.756 | 0.267 | 0.623 | 0.374 | 0.772 | 0.768 |
| Decision Tree (no depth limit) | 0.829 | 0.290 | 0.318 | 0.303 | 0.607 | 1.000 |
| **Random Forest (balanced)** | **0.850** | **0.401** | **0.578** | **0.474** | **0.802** | 0.915 |

The balanced **Random Forest** is the selected model. The unrestricted decision tree shows clear overfitting (train ROC-AUC 1.000 vs test 0.607).

## Key findings

- **Targeting value:** calling the model's **top 20%** of clients reaches **62%** of all subscribers, with a 36% success rate (about 3× the 11.7% base rate).
- Clients who said yes in a **previous campaign** subscribed at **64.7%**, about 5.5× the average.
- The youngest (≤25) and oldest (60+) clients respond best; clients with a housing loan respond less.
- Success falls from 14.6% after one call to 5.8% after six or more, though hesitant clients are also the ones called repeatedly.
- `contact = unknown` was the top feature but is likely a **data-recording artefact** of the May–June period.

## Limitations and next steps

Recording artefacts, class imbalance (precision about 40%), remaining overfitting, a single train/test split, missing customer data, and the age of the data (2008–2010). Next steps: threshold tuning, SMOTE, cross-validated hyperparameter search, gradient boosting, and retraining on recent campaigns.

## Files

- `bank_marketing_classification.ipynb`: full analysis with outputs
- `Bank_Marketing_Classification_Report.pdf`: stakeholder report

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib
