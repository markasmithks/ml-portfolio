# Cancer Risk Classification

## Overview
This project builds machine learning models to classify cancer risk levels
(**Low**, **Medium**, **High**) using demographic, lifestyle, genetic, and
environmental factors.

The goal is to demonstrate a clear, methodical machine learning workflow,
starting from a simple, interpretable baseline model and progressing to
more complex models.

---

## Data
- Format: CSV
- Target variable: `Risk_Code`
  - 0 = Low risk
  - 1 = Medium risk
  - 2 = High risk
- Identifier and leakage-related columns were removed prior to modeling.

---

## Model 1: Logistic Regression (Baseline)

Logistic regression was used as the baseline model due to its interpretability
and suitability for multiclass classification.

### Preprocessing
- Stratified train/test split (80/20)
- One-hot encoding for categorical features
- StandardScaler applied to numeric features
- Class weights balanced to account for class imbalance

### Results
- Accuracy: **91%**
- High-risk recall: **95%**
- Medium-risk recall: **81%**
- Macro F1-score: **0.90**

The model performs especially well at identifying high-risk cases, which is
critical in a medical risk classification context.

Logistic regression coefficients indicate that for this dataset High-risk predictions are most strongly driven by lifestyle and environmental exposure variables, including alcohol use (+1.69), air pollution (+1.65), diet (red meat +1.46, salted/processed +1.28), obesity (+1.20), and occupational hazards (+1.10). Genetic and medical history factors contribute moderately, while protective dietary factors such as fruit and vegetable intake reduce predicted risk (-0.54). Cancer type shows meaningful relative differences, with lung cancer associated with higher risk compared to the reference category (in this case the reference is Breast Cancer). Overall, the learned coefficients align well with known cancer risk factors, supporting the model’s basic validity.

---

## Project Structure

```text
CancerDataStudy/
├── data/
│   └── cancer-risk-factors.csv
├── src/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── gradient_boosting.py
├── notebooks/
│   
└── README.md
```


---



## Model 2: Random Forest

Random Forest was chosen as a second model to capture nonlinear interactions among lifestyle, environmental, and genetic risk factors that logistic regression cannot model.

### Preprocessing
The Random Forest model used the same preprocessing pipeline and train/test split as the logistic regression baseline to ensure a fair comparison:
- Stratified train/test split (80/20)
- One-hot encoding for categorical features
- StandardScaler applied to numeric features (retained for consistency, though not required for tree-based models)
- Class weights balanced to account for class imbalance

### Results
- Accuracy: **92%**
- High-risk recall: **95%**
- Medium-risk recall: **89%**
- Macro F1-score: **0.92**

This Random Forest model achieved a modest improvement in overall performance compared to logistic regression, with the most significant gain observed in Medium-risk classification. Importantly, High-risk recall remained unchanged, indicating that improved performance did not come at the expense of missing high-risk cases.

The confusion matrix shows that no High-risk cases were misclassified as Low risk, and only a single High-risk case was classified as Medium. Compared to logistic regression, Random Forest reduced misclassification of Medium-risk cases by better integrating multiple moderate risk factors, reflecting the model’s ability to capture nonlinear, multi-factor risk patterns.

Overall, these results suggest that while logistic regression provides valuable interpretability and insight into dominant risk drivers, Random Forest offers improved predictive performance for ambiguous, intermediate-risk cases by modeling interactions among features.

---

## Model 3: Gradient Boosting (Tree Boosting)

Gradient Boosting was added to evaluate whether sequentially optimized decision trees improve risk classification beyond bagged trees (Random Forest), particularly for subtle, intermediate-risk cases. This approach emphasizes correcting prior misclassifications and can produce sharper decision boundaries for complex tabular data.

Conceptually:
Logistic Regression → linear baseline
Random Forest → nonlinear averaging
Gradient Boosting → nonlinear + sequential error correction

### Preprocessing
The Gradient Boosting model used the same preprocessing pipeline and train/test split as the previous models:
- Stratified train/test split (80/20)
- One-hot encoding for categorical features
- StandardScaler applied to numeric features
- Class weights balanced to account for class imbalance

### Results
- Accuracy: **93%**
- High-risk recall: **95%**
- Medium-risk recall: **86%**
- Macro F1-score: **0.92**

The Gradient Boosting model returned  the highest overall accuracy among the evaluated models while maintaining strong High-risk recall. The confusion matrix indicates a conservative classification strategy: Medium-risk cases were more likely to be escalated to High risk rather than downgraded to Low risk, and no High-risk cases were misclassified as Low risk.

Compared to Random Forest, Gradient Boosting slightly increased overall accuracy but did not further improve Medium-risk recall. This suggests that while boosting sharpens decision boundaries, Random Forest provides a better balance for intermediate-risk classification in this dataset.

| Model               | Accuracy | Medium Recall | High Recall | Key Behavior          |
| ------------------- | -------- | ------------- | ----------- | --------------------- |
| Logistic Regression | 0.91     | 0.81          | 0.95        | Linear, interpretable |
| Random Forest       | 0.92     | **0.89**      | 0.95        | Best balance          |
| Gradient Boosting   | **0.93** | 0.86          | 0.95        | More conservative     |

## Model Comparison Summary
- Logistic Regression provides a strong, interpretable baseline and highlights dominant risk drivers.
- Random Forest offers the best balance between accuracy and Medium-risk classification by capturing nonlinear interactions.
- Gradient Boosting achieves the highest overall accuracy with a more conservative risk posture, favoring escalation over under-classification.

Together, these models demonstrate how increasing model complexity affects performance, interpretability, and risk sensitivity.

---

## Next Steps
- Compare Random Forest feature importance with logistic regression coefficients
- Evaluate model stability using cross-validation
- Explore calibrated probability thresholds for deployment-oriented use

---