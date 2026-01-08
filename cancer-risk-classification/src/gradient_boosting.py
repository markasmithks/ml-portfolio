from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report


# =====================================================
# 0. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "cancer-risk-factors.csv"


# =====================================================
# 1. Load data
# =====================================================

df = pd.read_csv(DATA_FILE)


# =====================================================
# 2. Target and features
# =====================================================

TARGET = "Risk_Code"

DROP_COLS = [
    "Risk_Level",
    "Overall_Risk_Score",
    "Patient_ID",
    "Row_ID",
]

X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = df[TARGET]


# =====================================================
# 3. Preprocessing
# =====================================================

categorical_cols = ["Cancer_Type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", numeric_transformer, numeric_cols),
    ]
)


# =====================================================
# 4. Gradient Boosting model
# =====================================================

gb_model = HistGradientBoostingClassifier(
    max_depth=5,
    learning_rate=0.05,
    max_iter=300,
    class_weight="balanced",
    random_state=42
)


# =====================================================
# 5. Pipeline
# =====================================================

gb_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", gb_model),
    ]
)


# =====================================================
# 6. Train / test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =====================================================
# 7. Train
# =====================================================

gb_pipeline.fit(X_train, y_train)


# =====================================================
# 8. Evaluate
# =====================================================

gb_preds = gb_pipeline.predict(X_test)

print("Gradient Boosting – Confusion Matrix:")
print(confusion_matrix(y_test, gb_preds))

print("\nGradient Boosting – Classification Report:")
print(classification_report(y_test, gb_preds))
