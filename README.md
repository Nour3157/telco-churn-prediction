# Telco Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn), using a decision tree built from scratch and a k-NN model from scikit-learn.

## What I did
- Loaded and summarized the dataset, then cleaned it (missing `TotalCharges` values filled with the training-set mean)
- Handled class imbalance by oversampling the training data only
- Implemented Gini index and entropy / information gain from scratch
- Built a decision tree from scratch (multi-way splits on categorical features, threshold splits on numeric features, max depth 6)
- Compared it against a k-NN model from scikit-learn
- Evaluated with accuracy, precision, recall, F1 and a cost-sensitive score (missing a churner costs 5x more than a false alarm)
- Ran 5-fold cross-validation and plotted learning curves

## Results (held-out test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Entropy tree | 73.5% | 50.0% | 79.1% | 0.613 |
| Gini tree | 72.6% | 49.0% | 80.4% | 0.609 |
| k-NN | 69.0% | 44.9% | 72.9% | 0.555 |

The from-scratch decision trees performed better than k-NN. Recall is the most important metric here, since the goal is to catch customers who are about to leave.

## Limitations
- The learning curve shows a gap between training and validation accuracy, which indicates overfitting.
- Cross-validation and learning curves were run on data that had already been oversampled, so duplicated minority-class rows can appear in both training and validation folds. Those scores are likely optimistic. The test-set results above are not affected, because the test set was split off before oversampling.
- Next steps: oversample only inside each training fold, and reduce the maximum tree depth.

## Data
Telco Customer Churn dataset (a public sample dataset, also available on Kaggle), stored in `data/telco_churn.csv`.

## How to run
1. Install the libraries: `pip install pandas numpy matplotlib scikit-learn`
2. Make sure `data/telco_churn.csv` is in the same folder as `main.py`
3. Run: `python main.py`

Plots are saved to an `outputs/` folder.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib


also includes: 
## Prototype: Customer Message Triage
Sorts customer messages into categories (billing, technical,
cancellation, general question, refund request, shipping problem) using a pretrained language model
(zero-shot classification, Hugging Face transformers).
Low-confidence predictions would be routed to a human.
Next step: combine with the churn model to draft retention messages.
