import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# -----------------------------
# 1. Load data
# -----------------------------

#df = pd.read_csv("C:\portfolio\CancerDataStudy\data\cancer-risk-factors.csv")
#df = pd.read_csv("data/cancer-risk-factors.csv")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cancer-risk-factors.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Define target and features
# -----------------------------
TARGET = "Risk_Code"

DROP_COLS = [
    "Risk_Level",          # human-readable label
    "Overall_Risk_Score",  # leakage
    "Patient_ID"
]


X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = df[TARGET]

# -----------------------------
# 3. Identify categorical features
# -----------------------------
categorical_cols = ["Cancer_Type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# -----------------------------
# 4. Preprocessing
# -----------------------------
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

# -----------------------------
# 5. Model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)

# -----------------------------
# 6. Pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# -----------------------------
# 7. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 8. Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# 9. Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------
# 10. Analyze Coefficients of Model
# ------------------------------------
logreg = pipeline.named_steps["model"]
logreg.coef_
logreg.intercept_
print(logreg.coef_.shape)
print(logreg.intercept_.shape)
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
coef_df = pd.DataFrame(
    logreg.coef_,
    columns=feature_names,
    index=["Low", "Medium", "High"]
)

print(coef_df.T.sort_values(by="High", ascending=False))