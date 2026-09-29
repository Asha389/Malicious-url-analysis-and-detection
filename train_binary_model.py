import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_PATH = "dataset/phishing.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("Original shape:", df.shape)

# ==========================================
# 2. REMOVE ROWS WITHOUT A TYPE
# ==========================================

df = df.dropna(subset=["type"]).copy()

print("Shape after removing missing types:", df.shape)

# ==========================================
# 3. CREATE BINARY TARGET
# ==========================================

# benign = SAFE = 0
# defacement/phishing/malware = MALICIOUS = 1

df["binary_label"] = df["type"].apply(
    lambda x: 0 if x == "benign" else 1
)

print("\nBinary label distribution:")
print(df["binary_label"].value_counts().sort_index())

# ==========================================
# 4. SELECT FEATURES
# ==========================================

# Columns that should NOT be used as model features
# because they directly identify the target or contain
# non-numeric/raw information.

excluded_columns = [
    "url",
    "type",
    "label",
    "binary_label",
    "Date_inspection"
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

X = df[feature_columns]
y = df["binary_label"]

# Keep only numeric columns
X = X.select_dtypes(include=["number"])

feature_columns = X.columns.tolist()

print("\nNumber of features:", len(feature_columns))
print("\nFeatures:")
print(feature_columns)

# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================

X = X.fillna(0)

# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 7. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training completed.")

# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 9. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==========================================")
print("BINARY MODEL RESULTS")
print("==========================================")

print(f"Accuracy :  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   :  {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score :  {f1:.4f} ({f1 * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Malicious"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 10. SAVE MODEL
# ==========================================

MODEL_PATH = "models/binary_malicious_url_model.pkl"
FEATURE_PATH = "models/binary_feature_columns.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURE_PATH)

print("\n==========================================")
print("MODEL SAVED")
print("==========================================")
print(MODEL_PATH)
print(FEATURE_PATH)